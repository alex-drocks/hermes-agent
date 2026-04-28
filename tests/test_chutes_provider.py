"""Tests for the Chutes.ai E2EE provider integration.

Covers provider registry entry, client construction with E2EE transport
injection, and graceful ImportError when `chutes-e2ee` is not installed.
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
    assert cfg.base_url_env_var == "CHUTES_BASE_URL"


# -----------------------------------------------------------------------------
# Models.dev Mapping
# -----------------------------------------------------------------------------


def test_chutes_in_models_dev_mapping():
    from agent.models_dev import PROVIDER_TO_MODELS_DEV

    assert PROVIDER_TO_MODELS_DEV.get("chutes") == "chutes"


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


class FakeAgent:
    """Minimal stand-in for AIAgent._create_openai_client dependencies."""

    provider = "chutes"

    @staticmethod
    def _build_keepalive_http_client(base_url: str = ""):  # pragma: no cover
        return None

    @staticmethod
    def _client_log_context():  # pragma: no cover
        return ""


def test_create_openai_client_chutes_import_error():
    """When chutes-e2ee is not installed, constructing a chutes client
    must raise RuntimeError with a clear install hint."""
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
    """When chutes-e2ee is available, the client must be constructed with
    an http_client backed by ChutesE2EETransport."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "chutes"

    mock_e2ee_transport_class = MagicMock()
    mock_e2ee_instance = MagicMock()
    mock_e2ee_transport_class.return_value = mock_e2ee_instance

    mock_httpx_client = MagicMock()
    mock_openai_client = MagicMock()

    with patch.dict(
        sys.modules,
        {"chutes_e2ee": MagicMock(ChutesE2EETransport=mock_e2ee_transport_class)},
    ):
        with patch("run_agent.OpenAI", return_value=mock_openai_client):
            with patch("httpx.Client", return_value=mock_httpx_client):
                result = agent._create_openai_client(
                    {"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"},
                    reason="test",
                    shared=False,
                )

    assert result is mock_openai_client
    # ChutesE2EETransport must have been instantiated with the correct args
    assert mock_e2ee_transport_class.call_count == 1
    call_kwargs = mock_e2ee_transport_class.call_args.kwargs
    assert call_kwargs["api_key"] == "cpk_test"
    assert call_kwargs["api_base"] == "https://api.chutes.ai"
    assert call_kwargs["models_base"] == "https://llm.chutes.ai"
    assert call_kwargs["inner"] is not None  # HTTPTransport with keepalive
