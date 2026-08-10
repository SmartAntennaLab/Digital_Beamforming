"""Compatibility facade for the digital-beamforming numerical API.

The implementation is intentionally divided by responsibility so numerical
code can evolve and be tested independently.  Existing callers may continue
to import from :mod:`beamforming`.
"""

from array_geometry import (
    ArrayCoordinates,
    GratingLobeAssessment,
    GratingLobeDirection,
    SteeringLimits,
    assess_grating_lobes,
    create_array_coordinates,
    create_array_taper,
    create_failure_mask,
    get_steering_limits,
    get_window_weights,
)
from array_math import (
    direction_cosines,
    element_pattern_factor,
    great_circle_directions,
    parse_phase_bits,
    quantize_phases,
    steering_phases,
    steering_vector,
)
from null_solver import (
    BeamformingWeights,
    ConstraintDiagnostics,
    compute_beamforming_weights,
)
from pattern_metrics import (
    ArrayGainMetrics,
    PatternMetrics,
    array_factor,
    calculate_array_gain_metrics,
    calculate_pattern_metrics,
    find_first_null,
    normalize_pattern_db,
    normalize_pattern_linear,
)

__all__ = [
    "ArrayCoordinates",
    "ArrayGainMetrics",
    "BeamformingWeights",
    "ConstraintDiagnostics",
    "GratingLobeAssessment",
    "GratingLobeDirection",
    "PatternMetrics",
    "SteeringLimits",
    "array_factor",
    "assess_grating_lobes",
    "calculate_array_gain_metrics",
    "calculate_pattern_metrics",
    "compute_beamforming_weights",
    "create_array_coordinates",
    "create_array_taper",
    "create_failure_mask",
    "direction_cosines",
    "element_pattern_factor",
    "find_first_null",
    "get_steering_limits",
    "get_window_weights",
    "great_circle_directions",
    "normalize_pattern_db",
    "normalize_pattern_linear",
    "parse_phase_bits",
    "quantize_phases",
    "steering_phases",
    "steering_vector",
]
