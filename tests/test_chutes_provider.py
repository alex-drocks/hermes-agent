"""Tests for the Chutes.ai E2EE provider integration.

Covers provider registry entry, canonical TUI list, client construction with
E2EE transport injection, and graceful ImportError when ``chutes-e2ee`` is
not installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Provider Registry
# -----------------------------------------------------------------------------


def test_chutes_in_provider_registry():
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert "chutes" in PROVIDER_REGISTRY
    cfg = PROVIDER_REGISTRY["chutes"]
    assert cfg.id == "chutes"
    assert cfg.name == "Chutes.ai"
    assert cfg.auth_type == "api_key"
    assert cfg.inference_base_url == "https://llm.chutes.ai/v1"
    assert cfg.api_key_env_vars == ("CHUTES_API_KEY",)
    # Strict endpoint — no base_url_env_var override
    assert cfg.base_url_env_var == ""


# -----------------------------------------------------------------------------
# Canonical Provider List (TUI)
# -----------------------------------------------------------------------------


def test_chutes_in_canonical_providers():
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS

    slugs = [p.slug for p in CANONICAL_PROVIDERS]
    assert "chutes" in slugs
    assert _PROVIDER_LABELS.get("chutes") == "Chutes.ai"


def test_chutes_in_list_available_providers():
    from hermes_cli.models import list_available_providers

    providers = list_available_providers()
    slugs = {p["id"] for p in providers}
    assert "chutes" in slugs


# -----------------------------------------------------------------------------
# Static Model Catalog & Live Discovery
# -----------------------------------------------------------------------------


def test_chutes_static_models():
    from hermes_cli.models import _PROVIDER_MODELS

    assert "chutes" in _PROVIDER_MODELS
    models = _PROVIDER_MODELS["chutes"]
    assert "default" in models
    assert "default:latency" in models
    assert "default:throughput" in models


def test_fetch_chutes_models_live(monkeypatch):
    """Live endpoint returns only TEE / confidential_compute=True models."""
    import json as _json

    from hermes_cli.models import fetch_chutes_models

    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")

    payload = {
        "object": "list",
        "data": [
            {"id": "moonshotai/Kimi-K2.6-TEE", "confidential_compute": True},
            {"id": "zai-org/GLM-5.1-TEE", "confidential_compute": True},
            {"id": "deepseek-ai/DeepSeek-V3", "confidential_compute": False},
            {"id": "Qwen/Qwen3-32B", "confidential_compute": False},
            {"id": "unsloth/gemma-3-27b-it"},  # missing field => False
        ],
    }

    fake_resp = MagicMock()
    fake_resp.read.return_value = _json.dumps(payload).encode()
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_resp)
    fake_cm.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=fake_cm):
        result = fetch_chutes_models()

    assert result is not None
    assert "moonshotai/Kimi-K2.6-TEE" in result
    assert "zai-org/GLM-5.1-TEE" in result
    # Non-TEE models must be excluded
    assert "deepseek-ai/DeepSeek-V3" not in result
    assert "Qwen/Qwen3-32B" not in result
    assert "unsloth/gemma-3-27b-it" not in result
    # Routing aliases still appended
    for alias in ("default", "default:latency", "default:throughput"):
        assert alias in result


def test_fetch_chutes_models_filters_non_tee():
    """Only models with ``confidential_compute=True`` survive filtering."""
    import json as _json

    from hermes_cli.models import fetch_chutes_models

    payload = {
        "data": [
            {"id": "a/TEE-Model", "confidential_compute": True},
            {"id": "b/NonTEE-Model", "confidential_compute": False},
            {"id": "c/MissingField"},
            {"id": "", "confidential_compute": True},  # blank id skipped
            {"confidential_compute": True},  # no id skipped
        ],
    }

    fake_resp = MagicMock()
    fake_resp.read.return_value = _json.dumps(payload).encode()
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_resp)
    fake_cm.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=fake_cm):
        result = fetch_chutes_models()

    assert result is not None
    assert result[0] == "a/TEE-Model"
    assert "b/NonTEE-Model" not in result
    assert "c/MissingField" not in result


def test_fetch_chutes_models_failure_falls_back(monkeypatch):
    """When the endpoint is unreachable, return ``None`` (caller falls back)."""
    from hermes_cli.models import fetch_chutes_models

    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    with patch("urllib.request.urlopen", side_effect=Exception("network down")):
        result = fetch_chutes_models()
    assert result is None


def test_provider_model_ids_wires_to_fetch_chutes(monkeypatch):
    """``provider_model_ids('chutes')`` routes to ``fetch_chutes_models``."""
    import json as _json

    from hermes_cli.models import provider_model_ids

    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    payload = {
        "data": [
            {"id": "deepseek-ai/DeepSeek-V3.2-TEE", "confidential_compute": True},
        ],
    }

    fake_resp = MagicMock()
    fake_resp.read.return_value = _json.dumps(payload).encode()
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_resp)
    fake_cm.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=fake_cm):
        result = provider_model_ids("chutes")

    assert "deepseek-ai/DeepSeek-V3.2-TEE" in result
    for alias in ("default", "default:latency", "default:throughput"):
        assert alias in result


# -----------------------------------------------------------------------------
# Aliases
# -----------------------------------------------------------------------------


def test_chutes_ai_alias_models():
    from hermes_cli.models import _PROVIDER_ALIASES, normalize_provider

    assert _PROVIDER_ALIASES.get("chutes-ai") == "chutes"
    assert normalize_provider("chutes-ai") == "chutes"


def test_chutes_ai_alias_auth():
    from hermes_cli.auth import resolve_provider

    assert resolve_provider("chutes-ai") == "chutes"


def test_chutes_ai_alias_providers():
    from hermes_cli.providers import normalize_provider as np

    assert np("chutes-ai") == "chutes"


def test_auth_commands_normalizes_chutes_ai():
    from hermes_cli.auth_commands import _normalize_provider

    assert _normalize_provider("chutes-ai") == "chutes"
    assert _normalize_provider("chutes") == "chutes"
    # Empty input preserved
    assert _normalize_provider("") == ""
    assert _normalize_provider(None) == ""  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Hermes Overlay
# -----------------------------------------------------------------------------


def test_chutes_overlay():
    from hermes_cli.providers import HERMES_OVERLAYS

    overlay = HERMES_OVERLAYS["chutes"]
    assert overlay.transport == "openai_chat"
    assert overlay.base_url_override == "https://llm.chutes.ai/v1"


# -----------------------------------------------------------------------------
# Models.dev Mapping
# -----------------------------------------------------------------------------


def test_chutes_in_models_dev_mapping():
    from agent.models_dev import PROVIDER_TO_MODELS_DEV

    assert PROVIDER_TO_MODELS_DEV.get("chutes") == "chutes"


# -----------------------------------------------------------------------------
# Model Metadata
# -----------------------------------------------------------------------------


def test_chutes_in_provider_prefixes():
    from agent.model_metadata import _PROVIDER_PREFIXES

    assert "chutes" in _PROVIDER_PREFIXES
    assert "chutes-ai" in _PROVIDER_PREFIXES


def test_chutes_in_url_to_provider():
    from agent.model_metadata import _URL_TO_PROVIDER

    assert _URL_TO_PROVIDER.get("llm.chutes.ai") == "chutes"


def test_chutes_prefers_live_endpoint_metadata():
    from agent.model_metadata import _provider_prefers_live_endpoint_metadata

    assert _provider_prefers_live_endpoint_metadata("https://llm.chutes.ai/v1")
    assert _provider_prefers_live_endpoint_metadata("https://llm.chutes.ai/v1/")
    assert not _provider_prefers_live_endpoint_metadata("https://api.openai.com/v1")
    assert not _provider_prefers_live_endpoint_metadata("")


# -----------------------------------------------------------------------------
# pyproject.toml Optional Dependency
# -----------------------------------------------------------------------------


def test_chutes_extra_declared():
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    extras = data.get("project", {}).get("optional-dependencies", {})
    assert "chutes" in extras
    assert any("chutes-e2ee" in dep for dep in extras["chutes"])


# -----------------------------------------------------------------------------
# Client Construction — _create_openai_client
# -----------------------------------------------------------------------------


def test_create_openai_client_chutes_import_error():
    """When ``chutes-e2ee`` is not installed, constructing a chutes client
    must raise ``RuntimeError`` with a clear install hint."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "chutes"

    with patch.dict(sys.modules, {"chutes_e2ee": None}):
        with pytest.raises(RuntimeError, match="chutes-e2ee"):
            agent._create_openai_client(
                {"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"},
                reason="test",
                shared=False,
            )


def test_create_openai_client_chutes_injects_transport():
    """When ``chutes-e2ee`` is available, the client must be constructed with
    an ``http_client`` backed by ``ChutesE2EETransport``."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "chutes"

    mock_e2ee_transport_class = MagicMock()
    mock_e2ee_instance = MagicMock()
    mock_e2ee_transport_class.return_value = mock_e2ee_instance

    mock_httpx_client = MagicMock()
    mock_openai_client = MagicMock()

    fake_module = MagicMock()
    fake_module.ChutesE2EETransport = mock_e2ee_transport_class

    with patch.dict(sys.modules, {"chutes_e2ee": fake_module}):
        with patch("run_agent.OpenAI", return_value=mock_openai_client):
            with patch("httpx.Client", return_value=mock_httpx_client):
                result = agent._create_openai_client(
                    {"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"},
                    reason="test",
                    shared=False,
                )

    assert result is mock_openai_client

    # ChutesE2EETransport instantiated once with correct kwargs
    assert mock_e2ee_transport_class.call_count == 1
    _call_kwargs = mock_e2ee_transport_class.call_args.kwargs
    assert _call_kwargs["api_base"] == "https://api.chutes.ai"
    assert _call_kwargs["models_base"] == "https://llm.chutes.ai"
    # inner must be an httpx HTTPTransport (not None)
    assert _call_kwargs["inner"] is not None


# -----------------------------------------------------------------------------
# Doctor — API-key provider health check
# -----------------------------------------------------------------------------


def test_chutes_in_doctor_apikey_checks():
    """``hermes doctor`` must include Chutes in the API-key health-check list."""
    import hermes_cli.doctor as doctor_mod

    # The _apikey_providers list is built inside the doctor() function,
    # so inspect its source for the literal tuple entries.
    import inspect as _inspect
    source = _inspect.getsource(doctor_mod.doctor)
    assert '"CHUTES_API_KEY"' in source
    assert '"Chutes.ai"' in source
    assert '"https://llm.chutes.ai/v1/models"' in source


# -----------------------------------------------------------------------------
# Config — env var registry
# -----------------------------------------------------------------------------


def test_chutes_api_key_in_env_config():
    """``CHUTES_API_KEY`` must be in the managed env-var registry so
    ``hermes setup`` prompts for it and ``hermes config set`` has metadata."""
    from hermes_cli.config import _ENV_CONFIG

    assert "CHUTES_API_KEY" in _ENV_CONFIG
    entry = _ENV_CONFIG["CHUTES_API_KEY"]
    assert entry["password"] is True
    assert entry["category"] == "provider"
    assert "chutes.ai" in entry["url"].lower()
