"""
tracer.py
---------
Records every tool call and result so you can inspect the agent's reasoning.
Displayed in the Streamlit UI as an expandable "Agent trace" panel.
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
from datetime import datetime


@dataclass
class AgentStep:
    step_num: int
    action: str                  # "tool_call" | "final_answer"
    tool_name: Optional[str]
    tool_input: Optional[dict]
    tool_result: Optional[Any]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "action": self.action,
            "tool": self.tool_name,
            "input": self.tool_input,
            "result_preview": str(self.tool_result)[:300] if self.tool_result else None,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class AgentTracer:
    def __init__(self):
        self.steps: List[AgentStep] = []
        self._counter = 0

    def reset(self):
        self.steps = []
        self._counter = 0

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_result: Any,
        duration_ms: Optional[float] = None,
    ) -> AgentStep:
        self._counter += 1
        step = AgentStep(
            step_num=self._counter,
            action="tool_call",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=tool_result,
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        return step

    def log_final_answer(self, answer: str) -> AgentStep:
        self._counter += 1
        step = AgentStep(
            step_num=self._counter,
            action="final_answer",
            tool_name=None,
            tool_input=None,
            tool_result=answer,
        )
        self.steps.append(step)
        return step

    def summary(self) -> dict:
        tool_calls = [s for s in self.steps if s.action == "tool_call"]
        return {
            "total_steps": self._counter,
            "tool_calls": len(tool_calls),
            "tools_used": list({s.tool_name for s in tool_calls}),
        }
