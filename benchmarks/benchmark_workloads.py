"""Benchmark full result workloads beyond the directivity kernel."""

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

from compute_tasks import ViewComputeRequest, calculate_view  # noqa: E402
from simulation import SimulationConfig  # noqa: E402

WORKLOAD_NAMES = (
    "directivity",
    "surface",
    "multi_null",
    "automatic_scan",
    "advanced_models",
)


@dataclass(frozen=True)
class WorkloadBenchmarkResult:
    workload: str
    frames: int
    elements: int
    seconds: float
    peak_rss_delta_mib: float
    valid: bool
    detail: str


def _peak_rss_sampler(stop: threading.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop.wait(0.002):
        samples.append(process.memory_info().rss)
    samples.append(process.memory_info().rss)


def _null_constraints() -> tuple[tuple[float, float, float], ...]:
    return (
        (30.0, 0.0, 30.0),
        (-30.0, 5.0, 30.0),
        (45.0, -10.0, 25.0),
        (-45.0, 10.0, 25.0),
    )


def _execute_workload(name: str):
    if name == "directivity":
        config = SimulationConfig(
            vertical_count=16,
            horizontal_count=16,
            directivity_mode="exact",
        )
        result = calculate_view(ViewComputeRequest(config, 5.0, 3.0, "metrics"))
        valid = (
            result.directivity is not None
            and result.directivity.directivity_dbi is not None
        )
        detail = (
            f"{result.directivity.effective_mode} "
            f"{result.directivity.directivity_dbi:.3f} dBi"
            if valid and result.directivity is not None
            else "directivity unavailable"
        )
        return 1, 256, valid, detail

    if name == "surface":
        config = SimulationConfig(vertical_count=16, horizontal_count=16)
        result = calculate_view(
            ViewComputeRequest(
                config,
                12.0,
                7.0,
                "pattern",
                scan_mode="full_3d",
            )
        )
        valid = result.surface is not None and result.surface.pattern_db.size > 0
        detail = (
            f"surface {result.surface.pattern_db.shape}"
            if result.surface is not None
            else "surface unavailable"
        )
        return 1, 256, valid, detail

    if name == "multi_null":
        config = SimulationConfig(
            vertical_count=12,
            horizontal_count=12,
            enable_null_steering=True,
            null_constraints_deg=_null_constraints(),
            null_optimization_mode="amplitude_phase",
        )
        result = calculate_view(ViewComputeRequest(config, 0.0, 0.0, "metrics"))
        valid = len(result.interferer_comparisons) == len(_null_constraints())
        return (
            1,
            144,
            valid,
            f"{len(result.interferer_comparisons)} Null constraints",
        )

    if name == "automatic_scan":
        config = SimulationConfig(vertical_count=8, horizontal_count=8)
        last_result = None
        for elevation in (-10.0, 10.0):
            for azimuth in np.linspace(-30.0, 30.0, 3):
                last_result = calculate_view(
                    ViewComputeRequest(
                        config,
                        float(azimuth),
                        elevation,
                        "pattern",
                        scan_mode="preview_3d",
                        scanning=True,
                    )
                )
        valid = (
            last_result is not None
            and last_result.surface_sampling is not None
            and last_result.surface_sampling.quality == "preview"
        )
        return 6, 64, valid, "6 preview scan frames"

    if name == "advanced_models":
        config = SimulationConfig(
            vertical_count=8,
            horizontal_count=8,
            target_azimuth_deg=20.0,
            position_error_rms_wavelength=0.01,
            amplitude_error_rms_db=0.2,
            phase_error_rms_deg=2.0,
            mutual_coupling_db=-30.0,
            wideband_bandwidth_percent=10.0,
            near_field_focus_range_m=0.5,
            enable_channel_analysis=True,
            channel_snapshots=128,
            multipath_count=2,
            adaptive_beamforming_method="lcmv",
            enable_doa_estimation=True,
        )
        result = calculate_view(ViewComputeRequest(config, 20.0, 0.0, "metrics"))
        analysis = result.advanced_analysis
        valid = bool(
            analysis is not None
            and analysis.wideband is not None
            and analysis.near_field is not None
            and analysis.channel is not None
            and analysis.adaptive is not None
            and analysis.doa is not None
        )
        detail = (
            f"squint {analysis.wideband.maximum_squint_deg:.2f} deg, "
            f"LCMV {analysis.adaptive.output_sinr_db:.2f} dB"
            if valid and analysis is not None
            else "advanced analysis unavailable"
        )
        return 1, 64, valid, detail

    raise ValueError(f"Unsupported workload: {name}")


def run_workload(name: str) -> WorkloadBenchmarkResult:
    if name not in WORKLOAD_NAMES:
        raise ValueError(f"Unsupported workload: {name}")
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
    started_at = time.perf_counter()
    try:
        frames, elements, valid, detail = _execute_workload(name)
    finally:
        duration = time.perf_counter() - started_at
        stop.set()
        sampler.join()
    return WorkloadBenchmarkResult(
        workload=name,
        frames=frames,
        elements=elements,
        seconds=duration,
        peak_rss_delta_mib=(max(rss_samples) - baseline_rss) / (1024.0**2),
        valid=valid,
        detail=detail,
    )


def _markdown(results: list[WorkloadBenchmarkResult]) -> str:
    lines = [
        "| Workload | Frames | Elements | Time (s) | Peak RSS Δ (MiB) | Valid | Detail |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.workload} | {result.frames} | {result.elements:,} | "
            f"{result.seconds:.3f} | {result.peak_rss_delta_mib:.1f} | "
            f"{'yes' if result.valid else 'no'} | {result.detail} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOAD_NAMES,
        default=list(WORKLOAD_NAMES),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = [run_workload(name) for name in args.workloads]
    print(_markdown(results))
    if args.json is not None:
        args.json.write_text(
            json.dumps([asdict(item) for item in results], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
