"""
agent.py
--------
Core agentic loop using Claude's tool-use API.

Flow per turn:
  1. Append user message to conversation history
  2. Call Claude with tool schemas
  3. If Claude calls a tool → execute it, feed result back, repeat
  4. If Claude says end_turn → return the final text answer
"""

import os
import ast
import operator
import time
from typing import List, Dict, Any, Optional

import anthropic
from dotenv import load_dotenv

from retriever import HybridRetriever
from tools import TOOLS
from tracer import AgentTracer

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 2048
MAX_ITERATIONS = 10  # prevent infinite tool-call loops

SYSTEM_PROMPT = """You are a precise research assistant with access to a private document collection.

Rules:
- ALWAYS call search_documents before answering questions about document content. Never guess from memory.
- For multi-part questions, search multiple times with focused sub-queries.
- When you need to do arithmetic on numbers found in documents, use the calculate tool.
- Cite sources inline like: [Source: filename, page N]
- If information is not in the documents, say so honestly — never fabricate.
- Keep answers clear and well-structured. Use bullet points or headers for complex answers.
"""

# ---------------------------------------------------------------------------
# Safe math evaluator (no eval/exec)
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic expressions safely without exec/eval."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "round":
            return round(*[_eval(a) for a in node.args])
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return _eval(ast.parse(expression, mode="eval").body)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Runs the actual tool logic and returns a string result to Claude."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def execute(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_documents":
            return self._search(**tool_input)
        if tool_name == "calculate":
            return self._calculate(**tool_input)
        if tool_name == "list_sources":
            return self._list_sources()
        return f"Unknown tool: {tool_name}"

    def _search(self, query: str, top_k: int = 4) -> str:
        top_k = min(int(top_k), 8)
        results = self.retriever.retrieve(query, top_k=top_k)
        if not results:
            return f"No results found for query: '{query}'"
        return self.retriever.format_context(results)

    def _calculate(self, expression: str) -> str:
        try:
            result = safe_eval(expression)
            return f"{expression} = {round(result, 6)}"
        except Exception as e:
            return f"Calculation error: {e}"

    def _list_sources(self) -> str:
        all_chunks = self.retriever.vector_store.get_all()
        if not all_chunks:
            return "No documents indexed yet."
        sources = sorted({c["metadata"].get("source", "unknown") for c in all_chunks})
        return "Indexed sources:\n" + "\n".join(f"- {s}" for s in sources)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Manages the full Claude tool-use loop.
    Maintains conversation history for multi-turn chat.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        tracer: Optional[AgentTracer] = None,
    ):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.retriever = retriever or HybridRetriever()
        self.executor = ToolExecutor(self.retriever)
        self.tracer = tracer or AgentTracer()
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.history = []
        self.tracer.reset()

    def run(self, user_message: str) -> str:
        """Process one user turn. Maintains history for multi-turn chat."""
        self.tracer.reset()
        self.history.append({"role": "user", "content": user_message})

        for _ in range(MAX_ITERATIONS):
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.history,
            )

            # Add Claude's response to history
            self.history.append({"role": "assistant", "content": response.content})

            # Done — extract and return text
            if response.stop_reason == "end_turn":
                answer = " ".join(
                    b.text for b in response.content if hasattr(b, "text")
                ).strip()
                self.tracer.log_final_answer(answer)
                return answer

            # Tool calls — execute each and collect results
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    t0 = time.time()
                    result = self.executor.execute(block.name, block.input)
                    elapsed = round((time.time() - t0) * 1000, 1)

                    self.tracer.log_tool_call(
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_result=result,
                        duration_ms=elapsed,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                self.history.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            break

        return "I reached the maximum number of reasoning steps. Please try a more specific question."
