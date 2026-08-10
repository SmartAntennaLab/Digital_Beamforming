"""Draft/applied state helpers for the multi-interferer controls."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

DRAFT_NULL_COUNT_KEY = "_draft_null_count"


def render_null_count_control() -> int:
    """Render the immediate input-area count without applying it to simulation."""

    st.session_state.setdefault(
        DRAFT_NULL_COUNT_KEY,
        int(st.session_state.get("null_count", 1)),
    )
    return int(
        st.sidebar.number_input(
            "간섭원 수",
            min_value=1,
            max_value=8,
            step=1,
            key=DRAFT_NULL_COUNT_KEY,
            help=(
                "개수를 바꾸면 입력 영역만 즉시 추가·제거됩니다. 새 방향과 "
                "억압량은 '설정 적용 및 계산'을 누를 때 계산에 반영됩니다."
            ),
        )
    )


def apply_draft_null_count() -> None:
    """Promote the visible draft count to the persistent simulation setting."""

    st.session_state["null_count"] = int(
        st.session_state.get(
            DRAFT_NULL_COUNT_KEY,
            st.session_state.get("null_count", 1),
        )
    )


def applied_null_constraints(
    state: Mapping[str, object],
) -> list[tuple[float, float, float]]:
    """Read only the last applied interferer list from persistent widget state."""

    count = int(state.get("null_count", 1))
    constraints = [
        (
            float(state.get("null_azimuth", 30.0)),
            float(state.get("null_elevation", 0.0)),
            float(state.get("null_1_suppression_db", 40.0)),
        )
    ]
    for null_index in range(2, count + 1):
        constraints.append(
            (
                float(state[f"null_{null_index}_azimuth"]),
                float(state[f"null_{null_index}_elevation"]),
                float(state[f"null_{null_index}_suppression_db"]),
            )
        )
    return constraints


__all__ = [
    "DRAFT_NULL_COUNT_KEY",
    "applied_null_constraints",
    "apply_draft_null_count",
    "render_null_count_control",
]
