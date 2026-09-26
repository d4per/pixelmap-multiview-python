"""What a running reconstruction reports.

Events carry data, not prose: which pair is being mapped, how far through it, why one
was rejected. :attr:`Event.message` renders any of them as a line for a log, but its
wording is not part of the contract; the fields are.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

__all__ = [
    "Event",
    "Started",
    "StageProgress",
    "PairProgress",
    "PairMapped",
    "PairRejected",
    "ViewDropped",
    "Log",
    "Status",
]


@dataclasses.dataclass(frozen=True)
class Event:
    """Something that happened while reconstructing.

    Attributes:
        stage: The stage it came from, one of the names in :data:`STAGES`.
        progress: How far through the whole run this is, ``0.0``-``1.0``, with each stage
            weighted by its typical cost. ``None`` for events that say nothing about
            progress — a log line or a dropped view — which a progress bar should ignore
            rather than move backwards for.
        message: The event as one line for a log.
    """

    stage: str
    progress: Optional[float]
    message: str


@dataclasses.dataclass(frozen=True)
class Started(Event):
    """The run has begun. ``pairs`` is how many pixelmap solves it will take."""

    views: int
    pairs: int


@dataclasses.dataclass(frozen=True)
class StageProgress(Event):
    """A stage reached a checkpoint, ``fraction`` of the way through it."""

    fraction: float


@dataclasses.dataclass(frozen=True)
class PairProgress(Event):
    """The solver advanced within one pair's mapping.

    ``pair`` is ``(a, b)`` with ``a < b``, the indices of the photos. ``index`` of ``of``
    is the pair's position in the run, from zero; ``step`` of ``steps`` is the solver's
    position within the pair.
    """

    pair: tuple[int, int]
    index: int
    of: int
    step: int
    steps: int


@dataclasses.dataclass(frozen=True, eq=False)
class PairMapped(Event):
    """One pair's dense correspondence is finished.

    Attributes:
        points: ``(rows, columns, 2)`` float32. ``points[r, c]`` is where the grid cell at
            ``(c * cell_size, r * cell_size)`` of photo ``a`` lands in photo ``b``, in the
            pixels of the photos handed in. ``NaN`` where the cell is unmapped.
        cell_size: Photo pixels between neighbouring grid cells.
        coverage: The fraction of photo ``a`` that is mapped.
    """

    pair: tuple[int, int]
    index: int
    of: int
    points: np.ndarray
    cell_size: float
    coverage: float


@dataclasses.dataclass(frozen=True)
class PairRejected(Event):
    """Two-view geometry cannot use a pair to place cameras.

    Its correspondences can still contribute to dense depth. ``reason`` is the kind, such
    as ``"planar"`` or ``"no_parallax"``; ``explanation`` says what it means for the
    photos.
    """

    pair: tuple[int, int]
    index: int
    of: int
    reason: str
    explanation: str


@dataclasses.dataclass(frozen=True)
class ViewDropped(Event):
    """A photo was left out of the reconstruction, and why."""

    view: int
    reason: str
    explanation: str


@dataclasses.dataclass(frozen=True)
class Log(Event):
    """Something worth saying that is not progress. ``level`` is ``"info"`` or
    ``"warning"``."""

    level: str


_KINDS = {
    "started": Started,
    "stage": StageProgress,
    "pair_progress": PairProgress,
    "pair_mapped": PairMapped,
    "pair_rejected": PairRejected,
    "view_dropped": ViewDropped,
    "log": Log,
}


def _event(kind: str, fields: dict) -> Event:
    """Builds the event the native module described as ``(kind, fields)``."""
    cls = _KINDS.get(kind, Event)
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{name: value for name, value in fields.items() if name in names})


@dataclasses.dataclass(frozen=True)
class Status:
    """Where a :class:`Job` has got to.

    Attributes:
        stage: The stage now running.
        progress: How far through the whole run, ``0.0``-``1.0``. Only ever moves forward.
        pairs_mapped: How many pairs have been mapped.
        pairs_total: How many there are to map; zero until the run has started.
        message: The most recent line worth showing.
    """

    stage: str
    progress: float
    pairs_mapped: int
    pairs_total: int
    message: str
