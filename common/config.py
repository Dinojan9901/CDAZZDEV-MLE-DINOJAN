import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GROQ_API_KEY = _get("GROQ_API_KEY")
GROQ_MODEL = _get("GROQ_MODEL", "openai/gpt-oss-120b")

OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TICKER = _get("DEFAULT_TICKER", "NVDA")

TASK1_OUTPUT_DIR = REPO_ROOT / "task1_financial" / "outputs"
TASK3_LOG_DIR = REPO_ROOT / "task3_agentic" / "logs"
TASK3_CACHE_DIR = REPO_ROOT / "task3_agentic" / "cache"

for _d in (TASK1_OUTPUT_DIR, TASK3_LOG_DIR, TASK3_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def available_providers() -> list[str]:
    providers = []
    if GROQ_API_KEY:
        providers.append("groq")
    if OPENROUTER_API_KEY:
        providers.append("openrouter")
    return providers


def require_llm_key() -> None:
    if not available_providers():
        raise RuntimeError(
            "No LLM key found. Copy .env.example to .env and set GROQ_API_KEY "
            "or OPENROUTER_API_KEY."
        )
