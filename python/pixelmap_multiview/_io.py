"""Reading photos from disc, which the Rust crate deliberately leaves to its callers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

__all__ = ["load_photos"]

_EXIF_IFD = 0x8769
_FOCAL_LENGTH_IN_35MM_FILM = 0xA405

PathArg = Union[str, "os.PathLike[str]"]


def load_photos(
    paths: Sequence[PathArg], *, max_size: Optional[int] = 1200
) -> tuple[list[np.ndarray], Optional[float]]:
    """Reads photos for :func:`reconstruct`, with the focal length EXIF records.

    Each photo is turned upright as its EXIF orientation says, converted to RGB, and
    scaled down so that its longer side is at most ``max_size`` pixels. Smaller photos
    take less memory, but the geometry that follows is less precise, so ``max_size``
    trades memory against detail. pixelmap scales photos down to a working width (400 px
    at ``"low"``, 800 at ``"medium"``, 1600 at ``"high"``, 3200 at ``"ultra"``) and never
    up, so photos narrower than that are matched at their own width: faster, but with a
    sparser mesh. Matching time therefore stays about the same at ``"low"`` and
    ``"medium"``, while ``"high"`` and ``"ultra"`` get slower and denser as the photos
    grow. Pass a larger ``max_size``, or ``None``, to give them the width to work at.

    Requires Pillow: ``pip install pixelmap-multiview-python[images]``.

    Args:
        paths: The photos, all from one camera at one zoom setting.
        max_size: The longest side to scale to, or ``None`` to keep full size.

    Returns:
        ``(photos, focal_35mm)``. ``focal_35mm`` is the 35 mm-equivalent focal length
        from EXIF when every photo records the same one, which is what
        :func:`reconstruct` takes as ``focal_35mm``; otherwise ``None``. The equivalent
        focal length does not depend on pixel count, so it stays right after scaling.

    Raises:
        ValueError: The photos are not all the same size once turned upright — say, one
            portrait among landscapes.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "load_photos needs Pillow: pip install pixelmap-multiview-python[images]"
        ) from err

    if max_size is not None and max_size < 1:
        raise ValueError(f"max_size must be positive, got {max_size}")

    photos: list[np.ndarray] = []
    focals = set()
    size = None
    for path in paths:
        with Image.open(path) as image:
            focals.add(image.getexif().get_ifd(_EXIF_IFD).get(_FOCAL_LENGTH_IN_35MM_FILM))
            image = ImageOps.exif_transpose(image).convert("RGB")

        if size is None:
            size = image.size
        elif image.size != size:
            raise ValueError(
                f"{os.fspath(path)!r} is {image.size[0]}x{image.size[1]} but the first "
                f"photo is {size[0]}x{size[1]}; all photos must come from one camera at "
                f"one resolution and orientation"
            )

        if max_size is not None and max(image.size) > max_size:
            scale = max_size / max(image.size)
            target = (round(image.size[0] * scale), round(image.size[1] * scale))
            image = image.resize(target, Image.Resampling.LANCZOS)
        photos.append(np.asarray(image))

    focal = focals.pop() if len(focals) == 1 else None
    return photos, (float(focal) if focal else None)
