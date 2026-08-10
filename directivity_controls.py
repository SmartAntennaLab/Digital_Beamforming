"""Directivity mode selection and large-array policy notices."""

from __future__ import annotations

import streamlit as st

from model_options import (
    DIRECTIVITY_MODE_LABELS,
    DIRECTIVITY_MODE_OPTIONS,
    option_label,
)
from resource_policy import ResourcePolicy, estimate_element_count


def render_directivity_mode_control() -> str:
    """Render the persisted exact/fast selection inside the settings form."""

    return st.selectbox(
        "Directivity 계산 모드",
        DIRECTIVITY_MODE_OPTIONS,
        format_func=lambda value: option_label(value, DIRECTIVITY_MODE_LABELS),
        key="directivity_mode",
        help=(
            "자동은 소자 수에 따라 정확/고속 모드를 선택합니다. 정확 모드는 "
            "O(N²) pairwise 적분, 고속 모드는 O(N×표본 수) 전구 적분입니다."
        ),
        persist_state="session",
    )


def render_directivity_policy_notice(
    policy: ResourcePolicy,
    *,
    mode: str,
    geometry: str,
    vertical_count: int,
    horizontal_count: int,
) -> None:
    """Explain warnings and automatic fallback before calculation starts."""

    element_count = estimate_element_count(
        geometry, vertical_count, horizontal_count
    )
    pair_count = element_count * element_count
    if mode == "exact" and element_count > policy.directivity_exact_max_elements:
        st.sidebar.warning(
            f"Directivity 정확 모드 상한 {policy.directivity_exact_max_elements:,}개를 "
            f"넘어 고속 근사로 전환됩니다 ({element_count:,}개 소자)."
        )
    elif mode == "exact" and element_count > policy.directivity_warning_elements:
        st.sidebar.warning(f"정확 Directivity가 {pair_count:,}개 pairwise 항을 계산합니다.")
    elif mode == "auto" and element_count > policy.directivity_warning_elements:
        st.sidebar.info(
            f"자동 모드가 {element_count:,}소자 배열에 고속 근사를 사용합니다."
        )
    elif mode == "fast":
        st.sidebar.caption("Directivity: 고속 근사 전구 적분을 사용합니다.")
