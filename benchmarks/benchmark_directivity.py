"""Measure wall time and peak RSS for large-array directivity modes."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from directivity import calculate_directivity  # noqa: E402


@dataclass(frozen=True)
class BenchmarkResult:
    array: str
    elements: int
    requested_mode: str
    effective_mode: str
    seconds: float
    peak_rss_delta_mib: float
    directivity_dbi: float | None
    samples: str


def _peak_rss_sampler(stop: threading.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop.wait(0.002):
        samples.append(process.memory_info().rss)
    samples.append(process.memory_info().rss)


def run_case(side: int, mode: str) -> BenchmarkResult:
    axis = (np.arange(side, dtype=float) - (side - 1) / 2.0) * 0.5
    y, z = np.meshgrid(axis, axis)
    weights = np.ones((side, side), dtype=complex)
    gc.collect()
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    rss_samples = [baseline_rss]
    stop = threading.Event()
    sampler = threading.Thread(
        target=_peak_rss_sampler,
        args=(stop, rss_samples),
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    try:
        result = calculate_directivity(
            y,
            z,
            weights,
            1.0,
            0.0,
            0.0,
            "isotropic",
            directivity_mode=mode,
        )
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join()
    peak_delta_mib = (max(rss_samples) - baseline_rss) / (1024.0**2)
    sample_text = (
        f"{result.azimuth_samples}x{result.elevation_samples}"
        if result.azimuth_samples is not None
        else "pairwise"
    )
    return BenchmarkResult(
        array=f"{side}x{side}",
        elements=side * side,
        requested_mode=mode,
        effective_mode=result.effective_mode,
        seconds=elapsed,
        peak_rss_delta_mib=max(0.0, peak_delta_mib),
        directivity_dbi=result.directivity_dbi,
        samples=sample_text,
    )


def _markdown(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Array | Elements | Requested | Effective | Time (s) | Peak RSS Δ (MiB) | Samples | Directivity (dBi) |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        directivity = (
            f"{item.directivity_dbi:.3f}"
            if item.directivity_dbi is not None
            else "N/A"
        )
        lines.append(
            f"| {item.array} | {item.elements:,} | {item.requested_mode} | "
            f"{item.effective_mode} | {item.seconds:.3f} | "
            f"{item.peak_rss_delta_mib:.1f} | {item.samples} | {directivity} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("auto", "exact", "fast"),
        default=["exact", "fast"],
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if any(side < 1 for side in args.sizes):
        parser.error("array sizes must be positive")

    results = [run_case(side, mode) for side in args.sizes for mode in args.modes]
    print(_markdown(results))
    if args.json is not None:
        args.json.write_text(
            json.dumps([asdict(item) for item in results], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
