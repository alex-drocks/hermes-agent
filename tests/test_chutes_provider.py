"""Tests for the Chutes.ai provider profile."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=response)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_chutes_profile_registers_metadata():
    from providers import get_provider_profile

    profile = get_provider_profile("chutes")

    assert profile is not None
    assert get_provider_profile("chutes-ai") is profile
    assert profile.display_name == "Chutes.ai"
    assert profile.base_url == "https://llm.chutes.ai/v1"
    assert profile.env_vars == ("CHUTES_API_KEY",)
    assert profile.fallback_models == (
        "Qwen/Qwen3-32B-TEE",
        "zai-org/GLM-5.1-TEE",
        "moonshotai/Kimi-K2.6-TEE",
    )
    assert profile.prefer_live_endpoint_metadata is True


def test_chutes_profile_auto_wires_cli_surfaces(monkeypatch):
    monkeypatch.delenv("CHUTES_API_KEY", raising=False)

    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.models import list_available_providers, provider_model_ids

    assert resolve_provider("chutes-ai") == "chutes"

    auth_config = PROVIDER_REGISTRY["chutes"]
    assert auth_config.name == "Chutes.ai"
    assert auth_config.inference_base_url == "https://llm.chutes.ai/v1"
    assert auth_config.api_key_env_vars == ("CHUTES_API_KEY",)

    env_config = OPTIONAL_ENV_VARS["CHUTES_API_KEY"]
    assert env_config["password"] is True
    assert env_config["category"] == "provider"
    assert env_config["url"] == "https://chutes.ai/app/api"

    listed = {entry["id"] for entry in list_available_providers()}
    assert "chutes" in listed
    with patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        return_value={"api_key": "", "base_url": "https://llm.chutes.ai/v1"},
    ):
        assert provider_model_ids("chutes") == [
            "Qwen/Qwen3-32B-TEE",
            "zai-org/GLM-5.1-TEE",
            "moonshotai/Kimi-K2.6-TEE",
        ]


def test_chutes_fetch_models_filters_to_confidential_compute(monkeypatch):
    from providers import get_provider_profile

    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    payload = {
        "data": [
            {"id": "moonshotai/Kimi-K2.6-TEE", "confidential_compute": True},
            {"id": "zai-org/GLM-5.1-TEE", "confidential_compute": True},
            {"id": "deepseek-ai/DeepSeek-V3", "confidential_compute": False},
            {"id": "Qwen/Qwen3-32B"},
            {"confidential_compute": True},
        ],
    }

    with patch("urllib.request.urlopen", return_value=_mock_json_response(payload)):
        result = get_provider_profile("chutes").fetch_models(api_key="cpk_test")

    assert result == [
        "moonshotai/Kimi-K2.6-TEE",
        "zai-org/GLM-5.1-TEE",
    ]


def test_chutes_fetch_models_failure_returns_none():
    from providers import get_provider_profile

    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        assert get_provider_profile("chutes").fetch_models(api_key="cpk_test") is None


def test_chutes_http_client_missing_dependency_has_install_hint():
    from providers import get_provider_profile

    with patch.dict(sys.modules, {"chutes_e2ee": None}):
        with patch("tools.lazy_deps.ensure", side_effect=RuntimeError("disabled")):
            with pytest.raises(RuntimeError, match="hermes-agent\\[chutes\\]"):
                get_provider_profile("chutes").build_http_client(api_key="cpk_test")


def test_chutes_http_client_uses_e2ee_transport():
    from providers import get_provider_profile

    e2ee_transport_class = MagicMock()
    e2ee_transport = MagicMock()
    e2ee_transport_class.return_value = e2ee_transport
    fake_module = MagicMock(ChutesE2EETransport=e2ee_transport_class)
    http_client = MagicMock()
    inner_transport = MagicMock()
    proxy_for_base_url = MagicMock(return_value="http://proxy.example")

    with patch.dict(sys.modules, {"chutes_e2ee": fake_module}):
        with patch("httpx.Client", return_value=http_client) as httpx_client:
            result = get_provider_profile("chutes").build_http_client(
                api_key="cpk_test",
                make_http_transport=MagicMock(return_value=inner_transport),
                proxy_for_base_url=proxy_for_base_url,
            )

    assert result is http_client
    e2ee_transport_class.assert_called_once_with(
        api_key="cpk_test",
        api_base="https://api.chutes.ai",
        inner=inner_transport,
    )
    proxy_for_base_url.assert_called_once_with("https://api.chutes.ai")
    httpx_client.assert_called_once_with(transport=e2ee_transport)


def test_chutes_http_client_applies_proxy_to_inner_transport():
    from providers import get_provider_profile

    e2ee_transport_class = MagicMock(return_value=MagicMock())
    fake_module = MagicMock(ChutesE2EETransport=e2ee_transport_class)
    http_client = MagicMock()
    inner_transport = MagicMock()
    make_http_transport = MagicMock(return_value=inner_transport)

    with patch.dict(sys.modules, {"chutes_e2ee": fake_module}):
        with patch("httpx.Client", return_value=http_client):
            get_provider_profile("chutes").build_http_client(
                api_key="cpk_test",
                make_http_transport=make_http_transport,
                proxy_for_base_url=MagicMock(return_value="http://proxy.example"),
            )

    make_http_transport.assert_called_once_with(proxy="http://proxy.example")
    e2ee_transport_class.assert_called_once_with(
        api_key="cpk_test",
        api_base="https://api.chutes.ai",
        inner=inner_transport,
    )


def test_create_openai_client_uses_provider_http_client_without_mutating_kwargs():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "chutes"
    agent.base_url = "https://llm.chutes.ai/v1"
    agent.model = "Qwen/Qwen3-32B-TEE"

    e2ee_transport_class = MagicMock(return_value=MagicMock())
    fake_module = MagicMock(ChutesE2EETransport=e2ee_transport_class)
    http_client = MagicMock()
    openai_client = MagicMock()
    kwargs = {"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"}

    with patch.dict(sys.modules, {"chutes_e2ee": fake_module}):
        with patch("httpx.Client", return_value=http_client):
            with patch("run_agent.OpenAI", return_value=openai_client) as openai:
                result = agent._create_openai_client(
                    kwargs,
                    reason="test",
                    shared=False,
                )

    assert result is openai_client
    assert kwargs == {"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"}
    openai.assert_called_once()
    assert openai.call_args.kwargs["http_client"] is http_client


def test_auxiliary_resolve_chutes_uses_e2ee_http_client():
    from agent.auxiliary_client import resolve_provider_client

    e2ee_transport_class = MagicMock(return_value=MagicMock())
    fake_module = MagicMock(ChutesE2EETransport=e2ee_transport_class)
    http_client = MagicMock()
    openai_client = MagicMock()

    with patch.dict(sys.modules, {"chutes_e2ee": fake_module}):
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"},
        ):
            with patch("httpx.Client", return_value=http_client):
                with patch("agent.auxiliary_client.OpenAI", return_value=openai_client) as openai:
                    client, model = resolve_provider_client(
                        "chutes",
                        "Qwen/Qwen3-32B-TEE",
                        async_mode=False,
                    )

    assert client is openai_client
    assert model == "Qwen/Qwen3-32B-TEE"
    openai.assert_called_once()
    assert openai.call_args.kwargs["http_client"] is http_client


def test_auxiliary_resolve_chutes_async_uses_async_e2ee_http_client():
    from agent.auxiliary_client import resolve_provider_client

    sync_transport_class = MagicMock(return_value=MagicMock())
    async_transport_class = MagicMock(return_value=MagicMock())
    fake_module = MagicMock(
        ChutesE2EETransport=sync_transport_class,
        AsyncChutesE2EETransport=async_transport_class,
    )
    sync_http_client = MagicMock()
    async_http_client = MagicMock()
    openai_client = MagicMock()
    openai_client.api_key = "cpk_test"
    openai_client.base_url = "https://llm.chutes.ai/v1"
    async_openai_client = MagicMock()

    with patch.dict(sys.modules, {"chutes_e2ee": fake_module}):
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "cpk_test", "base_url": "https://llm.chutes.ai/v1"},
        ):
            with patch("httpx.Client", return_value=sync_http_client):
                with patch("httpx.AsyncClient", return_value=async_http_client):
                    with patch("agent.auxiliary_client.OpenAI", return_value=openai_client):
                        with patch("openai.AsyncOpenAI", return_value=async_openai_client) as async_openai:
                            client, model = resolve_provider_client(
                                "chutes",
                                "Qwen/Qwen3-32B-TEE",
                                async_mode=True,
                            )

    assert client is async_openai_client
    assert model == "Qwen/Qwen3-32B-TEE"
    async_openai.assert_called_once()
    assert async_openai.call_args.kwargs["http_client"] is async_http_client
