"""Tests for OpenRouter routing in applypilot.llm and applypilot.config."""

import pytest

from applypilot import config, llm


@pytest.fixture(autouse=True)
def reset_llm_singleton() -> None:
    """Ensure each test starts with a fresh LLMClient singleton."""
    llm._instance = None
    yield
    llm._instance = None


def test_detect_provider_returns_openrouter_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter wins over every other provider when OPENROUTER_API_KEY is set."""
    monkeypatch.setattr(llm, "_OPENROUTER_KEY", "sk-or-test-key")
    monkeypatch.setattr(llm, "_GEMINI_KEY", "gemini-should-be-ignored")
    monkeypatch.setattr(llm, "_OPENAI_KEY", "openai-should-be-ignored")
    monkeypatch.setattr(llm, "_LOCAL_URL", "http://should-be-ignored")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    base_url, model, api_key = llm._detect_provider()

    assert base_url == "https://openrouter.ai/api/v1"
    assert model == "anthropic/claude-sonnet-4.6"
    assert api_key == "sk-or-test-key"


def test_detect_provider_openrouter_respects_llm_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_MODEL env var overrides the default OpenRouter model."""
    monkeypatch.setattr(llm, "_OPENROUTER_KEY", "sk-or-test-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-opus-4-6")

    _base_url, model, _api_key = llm._detect_provider()

    assert model == "anthropic/claude-opus-4-6"


def test_detect_provider_falls_back_to_gemini_when_openrouter_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing Gemini branch still works when OPENROUTER_API_KEY is unset."""
    monkeypatch.setattr(llm, "_OPENROUTER_KEY", "")
    monkeypatch.setattr(llm, "_GEMINI_KEY", "gemini-key")
    monkeypatch.setattr(llm, "_OPENAI_KEY", "")
    monkeypatch.setattr(llm, "_LOCAL_URL", "")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    base_url, model, api_key = llm._detect_provider()

    assert "generativelanguage.googleapis.com" in base_url
    assert model == "gemini-2.0-flash"
    assert api_key == "gemini-key"
