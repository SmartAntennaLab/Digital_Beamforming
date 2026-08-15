"""Browser-local settings persistence and share-link lifecycle."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from device_settings import (
    DEFAULT_DEVICE_SETTINGS,
    DEVICE_SETTING_KEYS,
    collect_device_settings,
    decode_share_token,
    encode_share_token,
    sanitize_device_settings,
    settings_envelope,
)
from device_storage import mount_device_storage


def apply_persistent_settings(settings: Mapping[str, object]) -> None:
    """Hydrate widget state before any persistent widget is instantiated."""

    sanitized = sanitize_device_settings(settings)
    for key, value in sanitized.items():
        st.session_state[key] = value
    if "enable_null" in sanitized:
        st.session_state["_draft_enable_null"] = bool(sanitized["enable_null"])
    if "null_count" in sanitized:
        st.session_state["_draft_null_count"] = int(sanitized["null_count"])


def next_storage_command(action: str, payload: object | None = None) -> None:
    """Queue one idempotent browser-storage command for the next rerun."""

    previous = st.session_state.get("_device_storage_command", {})
    command_id = int(previous.get("id", 0)) + 1
    command: dict[str, object] = {"id": command_id, "action": action}
    if payload is not None:
        command["payload"] = payload
    st.session_state["_device_storage_command"] = command


def request_device_settings_save() -> None:
    """Save the submitted widget state in this browser only."""

    settings = collect_device_settings(st.session_state)
    next_storage_command("save", settings_envelope(settings))
    st.session_state["_device_settings_applied"] = True
    if "settings" in st.query_params:
        st.query_params.clear()
        st.session_state["_applied_query_signature"] = None


def request_device_settings_clear() -> None:
    """Clear browser storage and return persistent widgets to defaults."""

    for key in DEVICE_SETTING_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("_draft_enable_null", None)
    st.session_state.pop("_draft_null_count", None)
    st.session_state["is_scanning"] = False
    st.session_state["scan_idx"] = 0
    st.session_state["scan_show_last_frame"] = False
    st.session_state.pop("scan_last_azimuth", None)
    st.session_state.pop("scan_last_elevation", None)
    st.session_state["_device_settings_applied"] = True
    st.session_state["_applied_query_signature"] = None
    st.query_params.clear()
    next_storage_command("clear")


def request_share_link() -> None:
    """Put a validated snapshot in one explicit URL query parameter."""

    token = encode_share_token(collect_device_settings(st.session_state))
    st.query_params.clear()
    st.query_params["settings"] = token
    st.session_state["_applied_query_signature"] = f"share:{token}"
    st.session_state["_settings_notice"] = (
        "info",
        "현재 주소가 공유 링크로 갱신되었습니다. 주소창의 URL을 복사하세요.",
    )


def initialize_settings_storage() -> None:
    """Initialize session defaults, URL settings, and browser storage state."""
    # Session state is deliberately small; numerical arrays live in bounded caches.
    st.session_state.setdefault("is_scanning", False)
    st.session_state.setdefault("scan_idx", 0)
    st.session_state.setdefault("scan_completed", False)
    st.session_state.setdefault("scan_show_last_frame", False)
    st.session_state.setdefault(
        "_device_storage_command",
        {"id": 0, "action": "load"},
    )
    st.session_state.setdefault("_device_settings_applied", False)
    for setting_key, default_value in DEFAULT_DEVICE_SETTINGS.items():
        st.session_state.setdefault(setting_key, default_value)
    st.session_state.setdefault(
        "_draft_enable_null",
        bool(st.session_state["enable_null"]),
    )
    st.session_state.setdefault(
        "_draft_null_count",
        int(st.session_state["null_count"]),
    )

    # Explicit share links take precedence over this browser's stored defaults.
    share_token = st.query_params.get("settings")
    query_signature: str | None = None
    query_settings: dict[str, object] = {}
    if isinstance(share_token, str) and share_token:
        query_signature = f"share:{share_token}"
        query_settings = decode_share_token(share_token)
    else:
        legacy_query = {
            key: st.query_params.get(key)
            for key in DEVICE_SETTING_KEYS
            if key in st.query_params
        }
        if legacy_query:
            query_signature = f"legacy:{sorted(legacy_query.items())!r}"
            query_settings = sanitize_device_settings(legacy_query)

    if (
        query_signature is not None
        and st.session_state.get("_applied_query_signature") != query_signature
    ):
        if query_settings:
            apply_persistent_settings(query_settings)
            st.session_state["_device_settings_applied"] = True
            st.session_state["_settings_notice"] = (
                "info",
                "URL에서 시뮬레이터 설정을 불러왔습니다.",
            )
            if query_signature.startswith("legacy:"):
                st.query_params.clear()
        else:
            st.session_state["_settings_notice"] = (
                "warning",
                "공유 링크 설정이 손상되었거나 지원하지 않는 형식입니다.",
            )
        st.session_state["_applied_query_signature"] = query_signature

    storage_result = mount_device_storage(st.session_state["_device_storage_command"])
    loaded_settings = getattr(storage_result, "loaded_settings", None)
    if not st.session_state["_device_settings_applied"] and isinstance(
        loaded_settings, Mapping
    ):
        apply_persistent_settings(loaded_settings)
        st.session_state["_device_settings_applied"] = True

    storage_status = getattr(storage_result, "status", None)
    if isinstance(storage_status, Mapping):
        status_id = storage_status.get("id")
        if status_id != st.session_state.get("_device_storage_status_seen"):
            st.session_state["_device_storage_status_seen"] = status_id
            action = storage_status.get("action")
            if storage_status.get("ok") and action == "save":
                st.session_state["_settings_notice"] = (
                    "success",
                    "현재 설정을 이 기기의 브라우저에 저장했습니다.",
                )
            elif storage_status.get("ok") and action == "clear":
                st.session_state["_settings_notice"] = (
                    "success",
                    "이 기기에 저장된 설정을 삭제하고 기본값으로 초기화했습니다.",
                )
            elif not storage_status.get("ok"):
                st.session_state["_settings_notice"] = (
                    "warning",
                    "브라우저 저장소를 사용할 수 없습니다. 브라우저 개인정보 보호 설정을 확인하세요.",
                )


__all__ = [
    "apply_persistent_settings",
    "initialize_settings_storage",
    "next_storage_command",
    "request_device_settings_clear",
    "request_device_settings_save",
    "request_share_link",
]
