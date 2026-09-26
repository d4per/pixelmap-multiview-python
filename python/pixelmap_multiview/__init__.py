"""3D reconstruction from three or more photos.

Python bindings for the `pixelmap_multiview <https://crates.io/crates/pixelmap_multiview>`_
Rust crate. It maps every pair of photos with pixelmap's dense correspondence, works out
where each photo was taken from, and fuses every view into one textured mesh.

    >>> import pixelmap_multiview as pmv
    >>> photos, focal = pmv.load_photos(["a.jpg", "b.jpg", "c.jpg"])  # doctest: +SKIP
    >>> model = pmv.reconstruct(photos, focal_35mm=focal)              # doctest: +SKIP
    >>> model.save("statue.obj")                                       # doctest: +SKIP

Nothing about the result is metric: the distance between the first two cameras is the
unit of length. A capture that cannot give a trustworthy model raises a
:class:`ReconstructionError` that names the stage and says what to change about the
photos, rather than returning a plausible-looking wrong model.

Runs are deterministic: the same photos, quality and seed give the same model.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator, Sequence
from typing import Callable, Optional, Union

import numpy as np

from . import _pixelmap_multiview
from ._errors import (
    CancelledError,
    InvalidInputError,
    PixelmapMultiviewError,
    ReconstructionError,
)
from ._events import (
    Event,
    Log,
    PairMapped,
    PairProgress,
    PairRejected,
    StageProgress,
    Started,
    Status,
    ViewDropped,
    _event,
)
from ._io import load_photos
from ._model import Intrinsics, Model, PairReport, Pose, Texture

__all__ = [
    "reconstruct",
    "Job",
    "load_photos",
    "Quality",
    "Model",
    "Texture",
    "Intrinsics",
    "Pose",
    "PairReport",
    "Event",
    "Started",
    "StageProgress",
    "PairProgress",
    "PairMapped",
    "PairRejected",
    "ViewDropped",
    "Log",
    "Status",
    "PixelmapMultiviewError",
    "InvalidInputError",
    "ReconstructionError",
    "CancelledError",
    "MIN_VIEWS",
    "MIN_DIMENSION",
    "MIN_PAIR_COVERAGE",
    "DEFAULT_MAX_TEXTURE_SIZE",
    "STAGES",
    "__version__",
]

try:  # pragma: no cover - trivial, and absent only in a broken install
    from importlib.metadata import version as _version

    __version__ = _version("pixelmap-multiview-python")
except Exception:  # pragma: no cover
    __version__ = "unknown"

#: The fewest photos a reconstruction needs.
MIN_VIEWS: int = _pixelmap_multiview.MIN_VIEWS

#: The smallest photo pixelmap can work with, on each side.
MIN_DIMENSION: int = _pixelmap_multiview.MIN_DIMENSION

#: The fraction of a photo a pair's mapping must cover to count as linking the two.
MIN_PAIR_COVERAGE: float = _pixelmap_multiview.MIN_PAIR_COVERAGE

#: The largest texture atlas a reconstruction produces by default, in pixels on a side.
DEFAULT_MAX_TEXTURE_SIZE: int = _pixelmap_multiview.DEFAULT_MAX_TEXTURE_SIZE

#: The pipeline's stages in the order they run, as ``(name, description)``. ``name`` is
#: what :attr:`Event.stage` and :attr:`PixelmapMultiviewError.stage` hold.
STAGES: list[tuple[str, str]] = _pixelmap_multiview.STAGES


class Quality(str, enum.Enum):
    """How much work pixelmap puts into each pair of photos.

    The dominant cost of a run, since there are N(N-1)/2 pairs. One pair takes roughly
    0.6 s at ``LOW``, 2.3 s at ``MEDIUM`` and 5.8 s at ``HIGH``.
    """

    LOW = "low"
    """Fastest. The default."""

    MEDIUM = "medium"
    """Slower, and more accurate."""

    HIGH = "high"
    """Slowest, and likely the best result."""

    def __str__(self) -> str:
        return self.value


QualityArg = Union[Quality, str]


def _quality_name(quality: QualityArg) -> str:
    if isinstance(quality, Quality):
        return quality.value
    if isinstance(quality, str):
        return quality
    raise TypeError(
        f"quality must be a pixelmap_multiview.Quality or one of 'low', 'medium', "
        f"'high', got {type(quality).__name__}"
    )


def _as_image(image, name: str) -> np.ndarray:
    """Normalises anything array-like into a contiguous ``(H, W, 3|4)`` uint8 array.

    Accepts NumPy arrays, Pillow images, and anything else ``np.asarray`` understands.
    Grayscale input — ``(H, W)`` or ``(H, W, 1)`` — is broadcast to three channels.
    """
    array = np.asarray(image)

    if array.dtype != np.uint8:
        raise ValueError(
            f"{name} must be a uint8 array, got dtype {array.dtype}. "
            f"Scale floating-point images to 0-255 and cast, e.g. "
            f"(img * 255).astype('uint8')."
        )

    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), "
            f"got {array.shape}"
        )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] not in (3, 4):
        raise ValueError(
            f"{name} must have 1, 3 or 4 channels, got {array.shape[2]} "
            f"(shape {array.shape})"
        )

    return np.ascontiguousarray(array)


def _arguments(
    photos,
    quality: QualityArg,
    focal_35mm: Optional[float],
    focal_px: Optional[float],
    seed: Optional[int],
    refine_focal: bool,
    max_texture_size: int,
) -> tuple:
    """Checks and normalises the arguments :func:`reconstruct` and :class:`Job` share,
    in the order the native functions take them."""
    if isinstance(photos, np.ndarray) and photos.ndim == 4:
        photos = list(photos)
    photos = [_as_image(photo, f"photos[{i}]") for i, photo in enumerate(photos)]

    if focal_35mm is not None and focal_px is not None:
        raise ValueError("give focal_35mm or focal_px, not both")
    if focal_35mm is not None:
        focal = ("35mm", float(focal_35mm))
    elif focal_px is not None:
        focal = ("pixels", float(focal_px))
    else:
        focal = ("unknown", 0.0)
    if focal[0] != "unknown" and not (np.isfinite(focal[1]) and focal[1] > 0):
        raise ValueError(f"the focal length must be positive, got {focal[1]}")

    if seed is not None and not (0 <= seed < 2**64):
        raise ValueError(f"seed must fit in an unsigned 64-bit integer, got {seed}")
    if max_texture_size < 1:
        raise ValueError(f"max_texture_size must be positive, got {max_texture_size}")

    return (
        photos,
        _quality_name(quality),
        focal[0],
        focal[1],
        seed,
        bool(refine_focal),
        int(max_texture_size),
    )


def reconstruct(
    photos: Sequence,
    *,
    quality: QualityArg = Quality.LOW,
    focal_35mm: Optional[float] = None,
    focal_px: Optional[float] = None,
    seed: Optional[int] = None,
    refine_focal: bool = False,
    max_texture_size: int = DEFAULT_MAX_TEXTURE_SIZE,
    on_event: Optional[Callable[[Event], None]] = None,
) -> Model:
    """Reconstructs a textured mesh from photos of one scene.

    Blocks until done, which is seconds for a few small photos and minutes for a
    walk-around of twenty. Use :class:`Job` to run it in the background instead.

    Args:
        photos: At least :data:`MIN_VIEWS` photos of a static scene, all from one camera
            at one zoom setting and all the same size, each a uint8 array of shape
            ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``. Pillow images work directly;
            :func:`load_photos` reads files. Step sideways between shots rather than
            turning on the spot, and let neighbouring photos overlap generously.
        quality: :class:`Quality` preset, or ``"low"``, ``"medium"`` or ``"high"``.
        focal_35mm: The 35 mm-equivalent focal length, as EXIF records it.
        focal_px: The focal length in pixels of the photos as handed in, for a calibrated
            camera. Give at most one of the two. With neither, the focal length is
            guessed and refined during bundle adjustment.
        seed: Seeds every random choice. The default is the crate's fixed seed, so runs
            are reproducible without asking.
        refine_focal: Refine a given focal length during bundle adjustment. Always done
            when none is given.
        max_texture_size: The largest texture atlas to produce, in pixels on a side.
        on_event: Called with an :class:`Event` at every checkpoint. An exception it
            raises stops the run and is re-raised here. Ctrl-C stops the run the same way,
            with or without a callback.

    Returns:
        A :class:`Model`.

    Raises:
        InvalidInputError: Too few photos, photos of different sizes, or a photo smaller
            than :data:`MIN_DIMENSION` on a side.
        ReconstructionError: No trustworthy model could be built from these photos. The
            message says why, and what to change.
        ValueError: A photo is not a uint8 array of an accepted shape, or a setting is
            out of range.
    """
    callback = None
    if on_event is not None:

        def callback(kind: str, fields: dict) -> None:
            on_event(_event(kind, fields))

    arguments = _arguments(
        photos, quality, focal_35mm, focal_px, seed, refine_focal, max_texture_size
    )
    return Model(_pixelmap_multiview.run(*arguments, callback))


class Job:
    """A reconstruction running on a background thread.

    Follow it by iterating, poll it with :attr:`status`, stop it with :meth:`cancel`, and
    collect the result with :meth:`join`::

        job = pmv.Job(photos, quality="medium")
        for event in job:
            print(f"{job.status.progress:.0%} {event.message}")
        model = job.join()

    Takes the same arguments as :func:`reconstruct`, apart from ``on_event``. The photos
    are checked for shape here; anything else wrong with them comes back from
    :meth:`join` as the exception :func:`reconstruct` would have raised.

    A job that is dropped without being joined is cancelled, rather than leaving a thread
    grinding through a reconstruction nobody is waiting for.
    """

    def __init__(
        self,
        photos: Sequence,
        *,
        quality: QualityArg = Quality.LOW,
        focal_35mm: Optional[float] = None,
        focal_px: Optional[float] = None,
        seed: Optional[int] = None,
        refine_focal: bool = False,
        max_texture_size: int = DEFAULT_MAX_TEXTURE_SIZE,
    ) -> None:
        arguments = _arguments(
            photos, quality, focal_35mm, focal_px, seed, refine_focal, max_texture_size
        )
        self._inner = _pixelmap_multiview.start(*arguments)

    def next_event(self, timeout: Optional[float] = None) -> Optional[Event]:
        """The next event, waiting at most ``timeout`` seconds for it.

        Returns ``None`` if the time runs out first. Raises ``StopIteration`` once the
        run is over and every event has been read. ``timeout=0`` polls without waiting,
        for a caller with its own event loop to keep turning.
        """
        described = self._inner.next_event(timeout)
        return None if described is None else _event(*described)

    def __iter__(self) -> Iterator[Event]:
        """Every event the run reports, in order, ending when the run does."""
        while True:
            try:
                yield self.next_event()
            except StopIteration:
                return

    @property
    def status(self) -> Status:
        """Where the run has got to, whether or not anyone is reading its events."""
        return Status(**self._inner.status())

    def cancel(self) -> None:
        """Asks the run to stop, and returns at once.

        The run ends at its next checkpoint — within a second even at high quality — and
        :meth:`join` then raises :class:`CancelledError`.
        """
        self._inner.cancel()

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._inner.cancelled

    def join(self) -> Model:
        """Waits for the run to finish and returns its model.

        Any events not yet read are discarded. Ctrl-C while waiting cancels the run.

        Raises:
            CancelledError: The run was stopped with :meth:`cancel`.
            InvalidInputError, ReconstructionError: As for :func:`reconstruct`.
        """
        return Model(self._inner.join())
