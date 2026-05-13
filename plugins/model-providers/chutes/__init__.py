"""Chutes.ai provider profile."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Callable

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

CHUTES_API_BASE = "https://api.chutes.ai"
CHUTES_MODELS_BASE = "https://llm.chutes.ai"
CHUTES_BASE_URL = f"{CHUTES_MODELS_BASE}/v1"
CHUTES_FALLBACK_MODELS = (
    "Qwen/Qwen3-32B-TEE",
    "zai-org/GLM-5.1-TEE",
    "moonshotai/Kimi-K2.6-TEE",
)
_USER_AGENT = f"HermesAgent/{_HERMES_VERSION}"


def _patch_model_discovery(transport: Any) -> None:
    """Point chutes-e2ee model discovery at Chutes' public model catalog.

    chutes-e2ee 0.1.0 assumes /v1/models and /e2e/* share one api_base.
    Chutes currently serves /v1/models from llm.chutes.ai and E2EE endpoints
    from api.chutes.ai, so the transport invoke base remains api.chutes.ai
    while model-name -> chute_id discovery reads llm.chutes.ai/v1/models.
    """
    discovery = getattr(transport, "_discovery", None)
    if discovery is None:
        return

    def _maybe_refresh_model_map(client: Any) -> None:
        now = time.time()
        if now - discovery._model_map_loaded_at < discovery._MODEL_MAP_TTL:
            return
        with discovery._model_map_lock:
            if now - discovery._model_map_loaded_at < discovery._MODEL_MAP_TTL:
                return
            resp = client.get(
                f"{CHUTES_MODELS_BASE}/v1/models",
                headers=discovery._auth_headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            discovery._model_map = {
                entry["id"]: entry["chute_id"]
                for entry in data
                if entry.get("id") and entry.get("chute_id")
            }
            discovery._model_map_loaded_at = time.time()

    discovery._maybe_refresh_model_map = _maybe_refresh_model_map


class ChutesProfile(ProviderProfile):
    """Chutes.ai OpenAI-compatible provider with E2EE transport."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return TEE-capable Chutes models."""
        url = self.models_url or self.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _USER_AGENT)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(chutes): %s", exc)
            return None

        items = data.get("data", []) if isinstance(data, dict) else data
        seen: set[str] = set()
        models: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not model_id or item.get("confidential_compute") is not True:
                continue
            if model_id not in seen:
                seen.add(model_id)
                models.append(model_id)
        return models

    def build_http_client(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "",
        make_http_transport: Callable[[], Any] | None = None,
        proxy_for_base_url: Callable[[str], Any] | None = None,
        **context: Any,
    ) -> Any | None:
        try:
            from chutes_e2ee import ChutesE2EETransport
        except ImportError:
            try:
                from tools.lazy_deps import ensure

                ensure("provider.chutes")
                from chutes_e2ee import ChutesE2EETransport
            except Exception as exc:
                raise RuntimeError(
                    "Chutes E2EE support requires `chutes-e2ee`. "
                    "Install with: pip install hermes-agent[chutes]"
                ) from exc

        import httpx

        inner = make_http_transport() if make_http_transport is not None else None
        if inner is None:
            inner = httpx.HTTPTransport()
        transport = ChutesE2EETransport(
            api_key=api_key or "",
            api_base=CHUTES_API_BASE,
            inner=inner,
        )
        _patch_model_discovery(transport)
        proxy = proxy_for_base_url(CHUTES_API_BASE) if proxy_for_base_url else None
        return httpx.Client(
            transport=transport,
            proxy=proxy,
        )


chutes = ChutesProfile(
    name="chutes",
    aliases=("chutes-ai",),
    display_name="Chutes.ai",
    description="Chutes.ai (TEE-secured E2EE inference)",
    signup_url="https://chutes.ai/app/api",
    env_vars=("CHUTES_API_KEY",),
    base_url=CHUTES_BASE_URL,
    models_url=f"{CHUTES_BASE_URL}/models",
    default_headers={"User-Agent": _USER_AGENT},
    fallback_models=CHUTES_FALLBACK_MODELS,
    prefer_live_endpoint_metadata=True,
)

register_provider(chutes)
