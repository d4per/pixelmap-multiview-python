"""Photos with known geometry, and one reconstruction of them shared by the suite.

The photos are renders of the crate's own synthetic scenes — the same ones its integration
tests reconstruct (`tests/progress.rs`, `tests/job.rs`) — so both suites judge the binding
against the same captures. A full run through the real matcher costs a few seconds, so it
happens once per session and the tests that only read the result share it.
"""

import pytest
from pixelmap_multiview import _pixelmap_multiview


def orbit(views: int, width: int, height: int, spread_deg: float = 36.0, scene="corner"):
    return _pixelmap_multiview._synthetic_orbit(scene, views, spread_deg, width, height)


@pytest.fixture(scope="session")
def photos():
    """Four views of two walls meeting at a corner, 640x480, over a 36 degree arc."""
    return orbit(4, 640, 480)


@pytest.fixture(scope="session")
def small_photos():
    """Three 200x150 views, for tests that stop the run rather than finish it."""
    return orbit(3, 200, 150, spread_deg=20.0)


@pytest.fixture(scope="session")
def recorded(photos):
    """The model of :func:`photos`, and every event its run reported."""
    import pixelmap_multiview as pmv

    events = []
    model = pmv.reconstruct(photos, on_event=events.append)
    return model, events


@pytest.fixture(scope="session")
def model(recorded):
    return recorded[0]


@pytest.fixture(scope="session")
def events(recorded):
    return recorded[1]
