"""Arguments are checked, and bad photos raise exceptions a caller can act on."""

import numpy as np
import pixelmap_multiview as pmv
import pytest


def test_too_few_photos_says_how_many_are_needed(photos):
    with pytest.raises(pmv.InvalidInputError) as caught:
        pmv.reconstruct(photos[:2])
    err = caught.value
    assert isinstance(err, ValueError), "input errors are ValueErrors too"
    assert err.reason == "too_few_photos"
    assert err.stage == "input"
    assert (err.found, err.minimum) == (2, pmv.MIN_VIEWS)


def test_photos_of_different_sizes_are_refused(photos):
    odd = photos[0][:400, :600]
    with pytest.raises(pmv.InvalidInputError) as caught:
        pmv.reconstruct([photos[0], photos[1], odd])
    err = caught.value
    assert err.reason == "size_mismatch"
    assert err.view == 2
    assert err.expected == (640, 480)
    assert err.found == (600, 400)


def test_a_photo_too_small_is_refused():
    tiny = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(pmv.InvalidInputError) as caught:
        pmv.reconstruct([tiny, tiny, tiny])
    assert caught.value.reason == "photo_too_small"
    assert caught.value.minimum == pmv.MIN_DIMENSION


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros((64, 64, 3), np.float32), "uint8"),
        (np.zeros((64, 64, 2), np.uint8), "channels"),
        (np.zeros((2, 64, 64, 3), np.uint8), "shape"),
    ],
)
def test_arrays_of_the_wrong_kind_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        pmv.reconstruct([bad, bad, bad])


def test_grey_and_rgb_photos_are_accepted_as_input(small_photos):
    # Checking the arguments is all that is asked of this: stop at the first event.
    class Stop(Exception):
        pass

    def stop(event):
        raise Stop

    grey = [p[..., 0] for p in small_photos]
    rgb = [p[..., :3] for p in small_photos]
    for photos in (grey, rgb):
        with pytest.raises(Stop):
            pmv.reconstruct(photos, on_event=stop)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"focal_35mm": 28.0, "focal_px": 900.0}, "not both"),
        ({"focal_35mm": -1.0}, "positive"),
        ({"focal_px": float("nan")}, "positive"),
        ({"seed": -1}, "64-bit"),
        ({"seed": 2**64}, "64-bit"),
        ({"max_texture_size": 0}, "max_texture_size"),
        ({"quality": "extreme"}, "extreme"),
    ],
)
def test_settings_out_of_range_are_refused(small_photos, kwargs, message):
    with pytest.raises(ValueError, match=message):
        pmv.reconstruct(small_photos, **kwargs)


def test_quality_must_be_a_string_or_preset(small_photos):
    with pytest.raises(TypeError):
        pmv.reconstruct(small_photos, quality=2)
