import json
import logging
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from common import config

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class SchemaValidationError(LLMError):
    pass


def _strip_fence(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return fenced.group(1).strip() if fenced else text.strip()


class LLMClient:
    """Groq first, OpenRouter on failure. Both speak an OpenAI-shaped chat API."""

    def __init__(self, temperature: float = 0.2, max_tokens: int = 1024,
                 model: str | None = None):
        config.require_llm_key()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model or config.GROQ_MODEL
        self.last_provider: str | None = None
        self._groq = None
        self._openrouter = None

        if config.GROQ_API_KEY:
            from groq import Groq

            self._groq = Groq(api_key=config.GROQ_API_KEY)
        if config.OPENROUTER_API_KEY:
            from openai import OpenAI

            self._openrouter = OpenAI(
                api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL
            )

    def _call(self, client, model, system, user, json_mode):
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs).choices[0].message.content

    # Groq's free tier allows 8000 tokens per minute, so a burst of calls trips the
    # limit even when every individual request is small. The window is 60 seconds, so
    # the backoff has to be long enough to outlast it rather than fail three times fast.
    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=3, min=3, max=45),
        reraise=True,
    )
    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        attempts = []
        if self._groq:
            attempts.append(("groq", self._groq, self.model))
        if self._openrouter:
            attempts.append(("openrouter", self._openrouter, config.OPENROUTER_MODEL))

        errors = []
        for name, client, model in attempts:
            try:
                content = self._call(client, model, system, user, json_mode)
                if not content:
                    raise ValueError("empty completion")
                self.last_provider = name
                return content
            except Exception as exc:
                log.warning("provider %s failed: %s", name, exc)
                errors.append(f"{name}: {exc}")

        raise LLMError("all providers failed -> " + " | ".join(errors))

    def structured(self, system: str, user: str, schema: Type[T], repair: bool = True) -> T:
        raw = self.chat(system, user, json_mode=True)
        try:
            return schema.model_validate_json(_strip_fence(raw))
        except (ValidationError, json.JSONDecodeError) as exc:
            log.warning("schema validation failed for %s: %s", schema.__name__, exc)
            if not repair:
                raise SchemaValidationError(str(exc)) from exc

        # One repair pass: hand the model its own bad output plus the error.
        fix_user = (
            f"{user}\n\nYour previous reply failed validation.\n"
            f"Previous reply:\n{raw}\n\nRequired JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n\n"
            "Return only corrected JSON."
        )
        retry_raw = self.chat(system, fix_user, json_mode=True)
        try:
            return schema.model_validate_json(_strip_fence(retry_raw))
        except (ValidationError, json.JSONDecodeError) as exc:
            log.error("repair pass failed for %s: %s", schema.__name__, exc)
            raise SchemaValidationError(str(exc)) from exc


_default: LLMClient | None = None


def get_client(**kwargs) -> LLMClient:
    global _default
    if _default is None or kwargs:
        client = LLMClient(**kwargs)
        if not kwargs:
            _default = client
        return client
    return _default
