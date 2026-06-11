"""Tests for QClaw public-gateway Agent routing (no relay on 19004)."""
from __future__ import annotations

from unittest.mock import patch

from pa_agent.ai.qclaw_connector import (
    _PUBLIC_GATEWAY_MODEL,
    _resolve_qclaw_endpoint,
    apply_qclaw_provider_to_settings,
    qclaw_provider_settings,
)


def test_resolve_qclaw_endpoint_skips_relay_by_default() -> None:
    with patch("pa_agent.ai.qclaw_connector._fetch_public_gateway_models") as fetch:
        fetch.return_value = ["openclaw", "openclaw/main"]
        with patch("pa_agent.ai.qclaw_relay_manager.ensure_qclaw_relay") as ensure:
            base_url, model, mode = _resolve_qclaw_endpoint(
                "127.0.0.1",
                58579,
                "token",
            )
    ensure.assert_not_called()
    assert base_url == "http://127.0.0.1:58579/v1"
    assert model == _PUBLIC_GATEWAY_MODEL
    assert "Agent" in mode


def test_qclaw_provider_settings_uses_openclaw_agent() -> None:
    with patch("pa_agent.ai.qclaw_connector._get_qclaw_gateway_info") as info:
        info.return_value = ("127.0.0.1", 58579, "secret")
        with patch("pa_agent.ai.qclaw_connector._resolve_qclaw_endpoint") as resolve:
            resolve.return_value = (
                "http://127.0.0.1:58579/v1",
                _PUBLIC_GATEWAY_MODEL,
                "公开网关（OpenClaw Agent / openclaw）",
            )
            settings = qclaw_provider_settings()
    assert settings is not None
    assert settings.model == _PUBLIC_GATEWAY_MODEL
    assert settings.base_url == "http://127.0.0.1:58579/v1"
    resolve.assert_called_once()
    assert resolve.call_args.kwargs.get("prefer_relay") is False


def test_apply_qclaw_provider_forces_agent_model() -> None:
    from pa_agent.config.settings import Settings

    settings = Settings()
    settings.provider.model = "openclaw"
    settings.provider.base_url = "http://127.0.0.1:1/v1"

    with patch(
        "pa_agent.ai.qclaw_connector.qclaw_provider_settings",
        return_value=type(
            "P",
            (),
            {
                "model": _PUBLIC_GATEWAY_MODEL,
                "base_url": "http://127.0.0.1:58579/v1",
                "api_key": "tok",
                "thinking": True,
                "reasoning_effort": "max",
                "context_window": 2_000_000,
            },
        )(),
    ):
        with patch("pa_agent.ai.qclaw_connector.detect_qclaw", return_value=True):
            with patch("pa_agent.ai.qclaw_connector.qclaw_health_check", return_value=(True, "ok")):
                err = apply_qclaw_provider_to_settings(settings)

    assert err is None
    assert settings.provider.model == _PUBLIC_GATEWAY_MODEL
    assert settings.provider.base_url == "http://127.0.0.1:58579/v1"
