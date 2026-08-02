"""Registry primitives for the detection ensemble.

The 14 detectors are deliberately heterogeneous: they expose detect(),
analyze() or extract(), they take images or catalogs or epoch lists, and
they have different preconditions. A common ABC is not possible without a
common signature, and renaming the public methods to force one would
touch a hundred call sites across the test suite to gain uniformity at a
single call site.

DetectorSpec adapts each detector instead. It records how to invoke one
detector and how to shape its output, so the ensemble runs a single
generic loop rather than fourteen hand-written blocks that drift apart.
The uniform interface is `run: Callable[[DetectionContext], dict]` -- a
bound closure per detector -- not a nominal base class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Exceptions a detector may raise on degenerate but legitimate input: an
# empty catalog, a frame with no sources, a fit that will not converge.
# These become an {"error": ...} entry so one bad detector cannot abort a
# whole region.
#
# Anything outside this tuple -- AttributeError from a typo, ImportError
# from a missing optional dependency, TypeError from a signature change --
# is a programming error and propagates. Catching those was how a broken
# detector could silently score zero for an entire run.
RECOVERABLE_DETECTOR_EXCEPTIONS: tuple[type[Exception], ...] = (
    ValueError,
    KeyError,
    IndexError,
    ZeroDivisionError,
    FloatingPointError,
    ArithmeticError,
    RuntimeError,
    np.linalg.LinAlgError,
)


@dataclass
class DetectionContext:
    """Everything the detectors share for a single detect() call.

    Detectors must treat this as read-only. Instances are built once and
    reused for every image in a run, and five of them execute concurrently
    in a thread pool, so a detector that stores per-image state on itself
    leaks that state into the next image and into its neighbours.
    """

    data: np.ndarray
    pixel_scale: float | None
    catalog: Any | None = None
    temporal_images: list | None = None
    positions: np.ndarray | None = None  # filled in after source extraction


@dataclass(frozen=True)
class DetectorSpec:
    """Declarative description of one ensemble member."""

    name: str
    """Key in the results dict, and the name of its enable gate."""

    run: Callable[[DetectionContext], dict[str, Any]]
    """Bound call into the detector. Raises are handled by the caller."""

    score_key: str
    """Key holding this detector's score in its own raw output."""

    weight_name: str
    """Key into config.ensemble_weights."""

    default_weight: float
    """Weight used when the config does not supply one."""

    summarize: Callable[[dict[str, Any]], dict[str, Any]]
    """Build the results entry from the raw output, in a fixed key order."""

    detail_keys: tuple[str, ...] = ()
    """Extra keys copied when truthy. Not for numpy arrays: truth-testing
    an array of more than one element raises."""

    detail_keys_if_present: tuple[str, ...] = ()
    """Extra keys copied when present, whatever their value. Use this for
    numpy arrays."""

    parallel: bool = False
    """Submit to the thread pool. Only worthwhile for image-heavy work
    that releases the GIL in C."""

    gated: bool = True
    """Honour the enable gate. Temporal has no gene, so it is ungated."""

    precondition: Callable[[DetectionContext], str | None] = lambda ctx: None
    """Return a skip-marker key, or None to run. Lets a detector report
    'no_catalog' rather than a zero that reads like a real measurement."""

    count_detections: Callable[[dict[str, Any]], int] = lambda raw: 0
    """This detector's contribution to results['n_detections']."""

    feature_index: int | None = None
    """Position in the anomaly detector's feature vector, if included."""

    extra: dict[str, Any] = field(default_factory=dict)
