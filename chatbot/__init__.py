"""VI Telecom Sales & Customer-Care Copilot -- chatbot package."""

from .catalog import get_catalog
from .prompt import build_system_instruction
from .llm import GeminiCopilot, get_api_key

__all__ = ["get_catalog", "build_system_instruction", "GeminiCopilot", "get_api_key"]
