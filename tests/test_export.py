"""Writing the model out for other tools, and reading photos in."""

import numpy as np
import pixelmap_multiview as pmv
import pytest
from PIL import Image


def test_obj_is_written_with_its_material_and_texture(model, tmp_path):
    files = model.save(tmp_path / "mesh.obj")
    assert [f.name for f in files] == ["mesh.obj", "mesh.mtl", "mesh_texture.png"]

    lines = (tmp_path / "mesh.obj").read_text().splitlines()
    assert "mtllib mesh.mtl" in lines
    assert sum(line.startswith("v ") for line in lines) == len(model.vertices)
    assert sum(line.startswith("f ") for line in lines) == len(model.triangles)
    assert "map_Kd mesh_texture.png" in (tmp_path / "mesh.mtl").read_text()

    with Image.open(tmp_path / "mesh_texture.png") as atlas:
        assert atlas.size == model.texture.atlas.shape[1::-1]
        assert np.array_equal(np.asarray(atlas.convert("RGBA")), model.texture.atlas)


def test_ply_has_a_colour_per_vertex(model, tmp_path):
    (path,) = model.save(tmp_path / "mesh.ply")
    header = path.read_text().split("end_header")[0]
    assert f"element vertex {len(model.vertices)}" in header
    assert f"element face {len(model.triangles)}" in header
    assert "property uchar red" in header


def test_x3d_refers_to_its_texture(model, tmp_path):
    files = model.save(tmp_path / "scene.x3d")
    assert [f.name for f in files] == ["scene.x3d", "scene_texture.png"]
    assert "scene_texture.png" in files[0].read_text()


def test_an_unknown_suffix_is_refused(model, tmp_path):
    with pytest.raises(ValueError, match=".obj"):
        model.save(tmp_path / "mesh.stl")


def test_load_photos_scales_and_reads_the_focal_length(photos, tmp_path):
    paths = []
    for i, photo in enumerate(photos[:3]):
        image = Image.fromarray(photo[..., :3])
        exif = image.getexif()
        exif.get_ifd(0x8769)[0xA405] = 28
        path = tmp_path / f"{i}.jpg"
        image.save(path, exif=exif)
        paths.append(path)

    loaded, focal = pmv.load_photos(paths, max_size=320)
    assert focal == 28.0
    assert [p.shape for p in loaded] == [(240, 320, 3)] * 3
    assert all(p.dtype == np.uint8 for p in loaded)


def test_load_photos_without_exif_has_no_focal_length(photos, tmp_path):
    paths = []
    for i, photo in enumerate(photos[:3]):
        path = tmp_path / f"{i}.png"
        Image.fromarray(photo).save(path)
        paths.append(path)
    loaded, focal = pmv.load_photos(paths, max_size=None)
    assert focal is None
    assert loaded[0].shape == (480, 640, 3)


def test_load_photos_refuses_mixed_sizes(photos, tmp_path):
    Image.fromarray(photos[0]).save(tmp_path / "a.png")
    Image.fromarray(photos[0][:300]).save(tmp_path / "b.png")
    with pytest.raises(ValueError, match="one resolution"):
        pmv.load_photos([tmp_path / "a.png", tmp_path / "b.png"])
