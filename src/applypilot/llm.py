"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment (highest precedence first):
  OPENROUTER_API_KEY  -> OpenRouter (default: anthropic/claude-sonnet-4.6)
  GEMINI_API_KEY      -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY      -> OpenAI (default: gpt-4o-mini)
  LLM_URL             -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
"""

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_LOCAL_URL = os.environ.get("LLM_URL", "")


def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables."""
    model_override = os.environ.get("LLM_MODEL", "")

    if _OPENROUTER_KEY:
        return (
            "https://openrouter.ai/api/v1",
            model_override or "anthropic/claude-sonnet-4.6",
            _OPENROUTER_KEY,
        )

    if _GEMINI_KEY and not _LOCAL_URL:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemini-2.0-flash",
            _GEMINI_KEY,
        )

    if _OPENAI_KEY and not _LOCAL_URL:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            _OPENAI_KEY,
        )

    if _LOCAL_URL:
        return (
            _LOCAL_URL.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_TIMEOUT = 120  # seconds


class LLMClient:
    """Thin OpenAI-compatible chat completions client using httpx."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    log.warning(
                        "LLM returned %s, retrying in %ds (attempt %d/%d)",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        # Should not reach here, but satisfy type checkers.
        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _instance
    if _instance is None:
        base_url, model, api_key = _detect_provider()
        log.info("LLM provider: %s  model: %s", base_url, model)
        _instance = LLMClient(base_url, model, api_key)
    return _instance
