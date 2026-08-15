"""Bounded MATLAB/measurement Golden Dataset import and cross-validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from element_pattern_data import evaluate_element_pattern
from pattern_metrics import array_factor

if TYPE_CHECKING:
    from simulation import SimulationState

GOLDEN_DATASET_SCHEMA_VERSION = 1
MAX_GOLDEN_FILE_BYTES = 1_000_000
MAX_GOLDEN_SAMPLES = 5_000


@dataclass(frozen=True)
class GoldenSample:
    azimuth_deg: float
    elevation_deg: float
    gain_db: float


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    source: str
    source_sha256: str
    normalization: str
    tolerance_db: float
    samples: tuple[GoldenSample, ...]


@dataclass(frozen=True)
class GoldenValidationResult:
    dataset_name: str
    source: str
    sample_count: int
    tolerance_db: float
    rmse_db: float
    maximum_error_db: float
    mean_error_db: float
    passed: bool


def _number(value, name: str, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _dataset(
    payload: bytes,
    *,
    name: str,
    source: str,
    normalization: str,
    tolerance_db: float,
    rows,
) -> GoldenDataset:
    if normalization != "peak_db":
        raise ValueError("Golden Dataset normalization must be 'peak_db'.")
    if source not in {"matlab", "measurement", "other"}:
        raise ValueError("Golden Dataset source must be matlab, measurement, or other.")
    samples: list[GoldenSample] = []
    for row_index, row in enumerate(rows, start=1):
        if len(samples) >= MAX_GOLDEN_SAMPLES:
            raise ValueError("Golden Dataset has too many samples.")
        try:
            samples.append(
                GoldenSample(
                    azimuth_deg=_number(
                        row["azimuth_deg"], "azimuth_deg", -180.0, 180.0
                    ),
                    elevation_deg=_number(
                        row["elevation_deg"], "elevation_deg", -90.0, 90.0
                    ),
                    gain_db=_number(row["gain_db"], "gain_db", -400.0, 200.0),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid Golden Dataset row {row_index}: {error}"
            ) from error
    if not samples:
        raise ValueError("Golden Dataset has no samples.")
    return GoldenDataset(
        name=name[:128],
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        normalization=normalization,
        tolerance_db=_number(tolerance_db, "tolerance_db", 0.0, 100.0),
        samples=tuple(samples),
    )


def parse_golden_dataset(payload: bytes, *, filename: str) -> GoldenDataset:
    """Parse a bounded UTF-8 CSV or JSON Golden Dataset."""

    if not payload:
        raise ValueError("Golden Dataset is empty.")
    if len(payload) > MAX_GOLDEN_FILE_BYTES:
        raise ValueError("Golden Dataset exceeds the 1 MB limit.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Golden Dataset must be UTF-8 encoded.") from error
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Golden Dataset JSON: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("Golden Dataset JSON root must be an object.")
        if document.get("schema_version") != GOLDEN_DATASET_SCHEMA_VERSION:
            raise ValueError("Unsupported Golden Dataset schema version.")
        return _dataset(
            payload,
            name=str(document.get("name") or filename),
            source=str(document.get("source") or "other").lower(),
            normalization=str(document.get("normalization") or "peak_db"),
            tolerance_db=document.get("tolerance_db", 1.0),
            rows=document.get("samples", ()),
        )
    if suffix != ".csv":
        raise ValueError("Golden Dataset must be a .csv or .json file.")
    reader = csv.DictReader(io.StringIO(text))
    required = {"azimuth_deg", "elevation_deg", "gain_db"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("Golden CSV requires azimuth_deg,elevation_deg,gain_db.")
    return _dataset(
        payload,
        name=filename,
        source="other",
        normalization="peak_db",
        tolerance_db=1.0,
        rows=reader,
    )


def validate_golden_dataset(
    state: SimulationState,
    dataset: GoldenDataset,
    *,
    cancel_check=None,
) -> GoldenValidationResult:
    """Compare normalized simulated gain against the imported references."""

    azimuth = np.radians([sample.azimuth_deg for sample in dataset.samples])
    elevation = np.radians([sample.elevation_deg for sample in dataset.samples])
    response = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        azimuth,
        elevation,
        cancel_check=cancel_check,
    )
    response *= evaluate_element_pattern(
        state.config.element_option,
        azimuth,
        elevation,
        pattern_grid=state.config.element_pattern_grid,
        polarization_angle_deg=state.config.polarization_angle_deg,
    )
    simulated_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
    reference_db = np.asarray([sample.gain_db for sample in dataset.samples])
    simulated_db -= np.max(simulated_db)
    reference_db -= np.max(reference_db)
    error = simulated_db - reference_db
    maximum_error = float(np.max(np.abs(error)))
    return GoldenValidationResult(
        dataset_name=dataset.name,
        source=dataset.source,
        sample_count=len(dataset.samples),
        tolerance_db=dataset.tolerance_db,
        rmse_db=float(np.sqrt(np.mean(error**2))),
        maximum_error_db=maximum_error,
        mean_error_db=float(np.mean(error)),
        passed=maximum_error <= dataset.tolerance_db,
    )


__all__ = [
    "GOLDEN_DATASET_SCHEMA_VERSION",
    "GoldenDataset",
    "GoldenSample",
    "GoldenValidationResult",
    "parse_golden_dataset",
    "validate_golden_dataset",
]
