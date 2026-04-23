"""
tools.py
--------
Tool schemas passed to Claude's API.
Claude reads these and decides which tool to call and with what arguments.
"""

TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the indexed document collection using a natural language query. "
            "Use this whenever the user asks about content that may be in the documents. "
            "Returns the most relevant text chunks with source citations. "
            "For complex questions, call this multiple times with different focused queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query. One concept per query works best.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return. Default 4, max 8.",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Safely evaluate a mathematical expression and return the result. "
            "Use this for arithmetic, percentages, growth rates, and ratios. "
            "Always retrieve numbers from documents first, then calculate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A Python arithmetic expression using numbers and operators (+, -, *, /, **, %). "
                        "Example: '(152 - 134) / 134 * 100'"
                    ),
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "list_sources",
        "description": (
            "List all document sources currently indexed in the knowledge base. "
            "Use this when the user asks what documents are available, "
            "or before summarising a specific file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
