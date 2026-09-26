"""A finished reconstruction."""

from __future__ import annotations

import dataclasses
import functools
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import _pixelmap_multiview

__all__ = ["Model", "Texture", "Intrinsics", "Pose", "PairReport"]


@dataclasses.dataclass(frozen=True, eq=False)
class Pose:
    """Where a camera is and which way it faces.

    Stored as the transform from world coordinates into the camera's frame,
    ``x_cam = rotation @ x_world + translation``. The camera frame is the usual
    computer-vision one: x to the right of the image, y down, z forward.
    """

    rotation: np.ndarray
    """``(3, 3)`` float64: world axes into camera axes."""

    translation: np.ndarray
    """``(3,)`` float64: the world origin, in the camera's frame."""

    @property
    def centre(self) -> np.ndarray:
        """The camera centre, in world coordinates: ``-rotation.T @ translation``."""
        return -self.rotation.T @ self.translation

    @property
    def matrix(self) -> np.ndarray:
        """The ``(3, 4)`` projection matrix ``[R | t]``."""
        return np.hstack([self.rotation, self.translation[:, np.newaxis]])

    @classmethod
    def _from(cls, fields: Optional[dict]) -> Optional[Pose]:
        return None if fields is None else cls(**fields)


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    """The pinhole camera every photo shares, in the pixels of the photos handed in."""

    fx: float
    fy: float
    cx: float
    cy: float
    source: str
    """Where the focal length came from: ``"provided"``, ``"exif35mm"``,
    ``"estimated"`` or ``"refined"``."""

    @property
    def matrix(self) -> np.ndarray:
        """The ``(3, 3)`` matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PairReport:
    """What two-view geometry made of one pair of photos.

    Attributes:
        pair: ``(a, b)``, the photos' indices, ``a < b``.
        coverage: The fraction of photo ``a`` its dense mapping covers.
        pose: Camera ``b`` in camera ``a``'s frame, with the distance between them scaled
            to 1. ``None`` when the pair was rejected.
        inlier_ratio: The share of sampled matches that agree on one camera motion.
        median_angle_deg: The median angle between the two cameras' rays, in degrees.
        rejection: Why the pair could not anchor a registration, such as ``"planar"``,
            or ``None`` if it could.
        explanation: What the rejection means for the photos.
    """

    pair: tuple[int, int]
    coverage: float
    pose: Optional[Pose]
    inlier_ratio: Optional[float]
    median_angle_deg: Optional[float]
    rejection: Optional[str]
    explanation: Optional[str]


class Texture:
    """Colour for a mesh: per vertex, and as a texture atlas.

    Each triangle's texture comes from the single photo that sees it best. Connected
    triangles that share a photo form a chart, and the charts are packed into one atlas.
    """

    def __init__(self, inner: _pixelmap_multiview.Model) -> None:
        self._inner = inner

    @functools.cached_property
    def atlas(self) -> np.ndarray:
        """``(H, W, 4)`` uint8 RGBA: the packed charts."""
        return self._inner.atlas

    @functools.cached_property
    def texcoords(self) -> np.ndarray:
        """``(K, 2)`` float32 coordinates into :attr:`atlas`, OBJ convention: ``v``
        grows upwards."""
        return self._inner.texcoords

    @functools.cached_property
    def face_texcoords(self) -> np.ndarray:
        """``(T, 3)`` uint32: for each triangle, indices into :attr:`texcoords` of its
        three corners. Separate from the vertex indices, because a vertex on a chart
        border has a different place in the atlas for each chart."""
        return self._inner.face_texcoords

    @functools.cached_property
    def face_views(self) -> np.ndarray:
        """``(T,)`` uint32: the photo each triangle's texture comes from."""
        return self._inner.face_views

    @property
    def charts(self) -> int:
        """How many charts the atlas holds."""
        return self._inner.charts

    @property
    def atlas_scale(self) -> float:
        """Atlas pixels per photo pixel; below 1 when the charts were shrunk to fit."""
        return self._inner.atlas_scale

    def __repr__(self) -> str:
        height, width = self.atlas.shape[:2]
        return (
            f"<pixelmap_multiview.Texture {width}x{height} atlas, {self.charts} charts>"
        )


class Model:
    """A finished reconstruction.

    Everything spatial is in one frame, which the best-connected pair of photos defines:
    its first camera sits at the origin looking along +z with +y down the image, and the
    distance between that pair's two cameras is the unit of length. **Nothing is
    metric.** :meth:`save` turns the result half a revolution about x so that viewers
    show it upright; the arrays here are not turned.

    Returned by :func:`reconstruct` and :meth:`Job.join`; not constructed directly.
    """

    def __init__(self, inner: _pixelmap_multiview.Model) -> None:
        if not isinstance(inner, _pixelmap_multiview.Model):
            raise TypeError(
                "Model is returned by pixelmap_multiview.reconstruct(), not constructed "
                "directly"
            )
        self._inner = inner

    # -- the mesh -------------------------------------------------------------

    @functools.cached_property
    def vertices(self) -> np.ndarray:
        """``(V, 3)`` float64 vertex positions."""
        return self._inner.vertices

    @functools.cached_property
    def normals(self) -> np.ndarray:
        """``(V, 3)`` float64 unit normals, pointing towards the side the cameras saw."""
        return self._inner.normals

    @functools.cached_property
    def triangles(self) -> np.ndarray:
        """``(T, 3)`` uint32 vertex indices, counter-clockwise seen from the side the
        normals point to."""
        return self._inner.triangles

    @functools.cached_property
    def vertex_colours(self) -> np.ndarray:
        """``(V, 3)`` uint8 RGB: each vertex a blend of the photos that see it best."""
        return self._inner.vertex_colours

    @functools.cached_property
    def texture(self) -> Texture:
        """The texture atlas and its coordinates."""
        return Texture(self._inner)

    # -- the cameras ----------------------------------------------------------

    @functools.cached_property
    def intrinsics(self) -> Intrinsics:
        """The camera the later stages used, with its focal length refined if it was."""
        return Intrinsics(**self._inner.intrinsics)

    @functools.cached_property
    def cameras(self) -> list[Optional[Pose]]:
        """Where each photo was taken from, by index. ``None`` for a photo left out."""
        return [Pose._from(fields) for fields in self._inner.cameras]

    @property
    def views(self) -> int:
        """How many photos went in."""
        return self._inner.views

    @functools.cached_property
    def connected(self) -> list[int]:
        """The indices of the photos that made it into the model."""
        return self._inner.connected

    @functools.cached_property
    def pairs(self) -> list[PairReport]:
        """Every pair's two-view geometry, including the pairs that were not usable."""
        reports = []
        for fields in self._inner.pairs:
            fields["pose"] = Pose._from(fields["pose"])
            reports.append(PairReport(**fields))
        return reports

    # -- writing out ----------------------------------------------------------

    def save(self, path: Union[str, os.PathLike]) -> list[Path]:
        """Writes the model for MeshLab, Blender, CloudCompare and the like.

        The format follows the suffix:

        - ``.obj``: textured Wavefront OBJ, with a ``.mtl`` material library and the atlas
          as ``<stem>_texture.png`` next to it.
        - ``.x3d``: textured X3D, with the atlas as ``<stem>_texture.png`` next to it.
        - ``.ply``: ASCII PLY with a colour per vertex, for viewers that do not load
          textures.

        The model is turned half a revolution about x so that viewers show it upright.

        Returns:
            Every file written, the one named first.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        texture = path.with_name(f"{path.stem}_texture.png")

        if suffix == ".obj":
            mtl = path.with_suffix(".mtl")
            files = {
                path: self._inner.obj(mtl.name),
                mtl: self._inner.mtl(texture.name),
                texture: self._inner.atlas_png(),
            }
        elif suffix == ".x3d":
            files = {
                path: self._inner.x3d(texture.name),
                texture: self._inner.atlas_png(),
            }
        elif suffix == ".ply":
            files = {path: self._inner.ply()}
        else:
            raise ValueError(
                f"cannot tell the format from {path.name!r}; use a .obj, .x3d or .ply "
                f"suffix"
            )

        for target, data in files.items():
            target.write_bytes(data)
        return list(files)

    def __repr__(self) -> str:
        return (
            f"<pixelmap_multiview.Model {len(self.vertices)} vertices, "
            f"{len(self.triangles)} triangles, "
            f"{len(self.connected)} of {self.views} photos>"
        )
