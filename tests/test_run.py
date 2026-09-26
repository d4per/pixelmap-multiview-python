"""A whole reconstruction, judged on what the model holds."""

import numpy as np
import pixelmap_multiview as pmv


def test_every_photo_is_placed(model):
    assert model.views == 4
    assert model.connected == [0, 1, 2, 3]
    assert len(model.cameras) == 4
    assert all(camera is not None for camera in model.cameras)


def test_the_first_camera_of_the_frame_sits_at_the_origin(model):
    # The seed pair defines the frame: one of its cameras is the identity, and the
    # distance between the two is the unit of length.
    at_origin = [c for c in model.cameras if np.allclose(c.centre, 0.0, atol=1e-9)]
    assert len(at_origin) == 1
    assert np.allclose(at_origin[0].rotation, np.eye(3))


def test_poses_are_rotations(model):
    for camera in model.cameras:
        assert camera.rotation.shape == (3, 3)
        assert camera.translation.shape == (3,)
        assert np.allclose(camera.rotation @ camera.rotation.T, np.eye(3), atol=1e-9)
        assert camera.matrix.shape == (3, 4)


def test_the_mesh_arrays_agree_with_each_other(model):
    v, t = model.vertices, model.triangles
    assert v.ndim == 2 and v.shape[1] == 3 and v.dtype == np.float64
    assert t.ndim == 2 and t.shape[1] == 3 and t.dtype == np.uint32
    assert len(v) > 1000 and len(t) > 1000
    assert t.max() < len(v)
    assert model.normals.shape == v.shape
    assert np.allclose(np.linalg.norm(model.normals, axis=1), 1.0, atol=1e-6)
    assert model.vertex_colours.shape == v.shape
    assert model.vertex_colours.dtype == np.uint8


def test_the_texture_covers_every_triangle(model):
    texture = model.texture
    assert texture.atlas.ndim == 3 and texture.atlas.shape[2] == 4
    assert max(texture.atlas.shape[:2]) <= pmv.DEFAULT_MAX_TEXTURE_SIZE
    assert texture.face_texcoords.shape == model.triangles.shape
    assert texture.face_texcoords.max() < len(texture.texcoords)
    assert texture.face_views.shape == (len(model.triangles),)
    assert set(np.unique(texture.face_views)) <= set(model.connected)
    assert texture.charts > 0


def test_intrinsics_describe_the_photos(model):
    k = model.intrinsics
    # No focal length was given, so it was guessed and then refined.
    assert k.source == "refined"
    assert (k.cx, k.cy) == (320.0, 240.0)
    assert k.matrix.shape == (3, 3)
    assert k.matrix[0, 0] == k.fx


def test_every_pair_is_reported(model):
    assert [r.pair for r in model.pairs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    for report in model.pairs:
        assert 0.0 <= report.coverage <= 1.0
        assert (report.pose is None) == (report.rejection is not None)


def test_the_same_seed_gives_the_same_model(photos, model):
    again = pmv.reconstruct(photos)
    assert np.array_equal(again.vertices, model.vertices)
    assert np.array_equal(again.triangles, model.triangles)


def test_a_given_focal_length_is_used(photos):
    model = pmv.reconstruct(photos, focal_px=640.0, max_texture_size=512)
    assert model.intrinsics.source == "provided"
    assert model.intrinsics.fx == 640.0
    assert max(model.texture.atlas.shape[:2]) <= 512


def test_model_repr_is_informative(model):
    assert "4 of 4 photos" in repr(model)
