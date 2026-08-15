"""Application version and source-control provenance without shell commands."""

from __future__ import annotations

import os
from pathlib import Path

APP_VERSION = "1.7.0"


def git_commit(project_root: Path | None = None) -> str:
    """Return an injected or locally resolved Git commit, if available."""

    injected = os.environ.get("DBF_GIT_COMMIT", "").strip()
    if injected:
        return injected[:64]
    root = project_root or Path(__file__).resolve().parent
    git_path = root / ".git"
    try:
        if git_path.is_file():
            pointer = git_path.read_text(encoding="utf-8").strip()
            git_path = (root / pointer.removeprefix("gitdir:").strip()).resolve()
        head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            reference = head.removeprefix("ref:").strip()
            loose = git_path / reference
            if loose.exists():
                return loose.read_text(encoding="utf-8").strip()[:64]
            packed = git_path / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(f" {reference}"):
                        return line.split(maxsplit=1)[0][:64]
        return head[:64]
    except (OSError, ValueError):
        return "unknown"


__all__ = ["APP_VERSION", "git_commit"]
