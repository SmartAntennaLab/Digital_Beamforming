"""Compatibility facade for Streamlit result renderers."""

from __future__ import annotations

import numpy as np
import streamlit as st

from simulation import SimulationState
from ui_elements import render_elements_tab
from ui_metrics import render_metrics_tab
from ui_pattern import pattern_figure, render_pattern_tab


def render_diagnostics(state: SimulationState) -> None:
    result = state.weight_result
    config = state.config
    if config.enable_null_steering and not result.null_applied:
        rank = (
            f"{result.constraint_rank}/{result.constraint_count}"
            if result.constraint_rank is not None
            else "N/A"
        )
        condition = (
            f"{result.condition_number:.3e}"
            if result.condition_number is not None
            and np.isfinite(result.condition_number)
            else "∞"
        )
        st.warning(
            "⚠️ 영점 제약 행렬이 특이하거나 수치적으로 불안정해 기본 조향 "
            f"가중치를 사용합니다. rank={rank}, condition={condition}."
        )
    elif config.enable_null_steering:
        unmet = [
            index + 1
            for index, status in enumerate(result.null_requirement_met)
            if status is False
        ]
        if unmet:
            directions = ", ".join(f"Null {index}" for index in unmet)
            st.warning(
                "⚠️ 최종 가중치가 요구 억압량을 충족하지 못했습니다: "
                f"{directions}. 진폭 상한, 위상 전용 모드 또는 위상 양자화 조건을 "
                "완화하세요."
            )

    assessment = state.grating_assessment
    if not assessment.has_aliasing_risk:
        return
    if assessment.risk_only:
        st.warning(
            "⚠️ UCA 인접 chord 간격이 0.5λ를 초과해 공간 앨리어싱 위험이 "
            "있습니다. UCA에는 직교 주기 배열의 복제 각도 식을 적용하지 않습니다."
        )
        return
    directions = ", ".join(
        f"(p,q)=({item.order_y},{item.order_z}) → "
        f"Az {item.azimuth_deg:.1f}°, El {item.elevation_deg:.1f}°"
        for item in assessment.directions
    )
    st.warning(f"⚠️ 가시 영역 격자 로브가 감지되었습니다: {directions}")


__all__ = [
    "pattern_figure",
    "render_diagnostics",
    "render_elements_tab",
    "render_metrics_tab",
    "render_pattern_tab",
]
