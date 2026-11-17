# pixelmap-multiview-python

[![PyPI](https://img.shields.io/pypi/v/pixelmap-multiview-python.svg)](https://pypi.org/project/pixelmap-multiview-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/pixelmap-multiview-python.svg)](https://pypi.org/project/pixelmap-multiview-python/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Turn a set of ordinary photos of a scene into a textured 3D model.**

Python bindings for the [`pixelmap_multiview`](https://crates.io/crates/pixelmap_multiview)
Rust crate. It maps every pair of three or more photos with
[pixelmap](https://pypi.org/project/pixelmap-python/)'s dense correspondence, works out
where each photo was taken from, and fuses all of the views into one mesh. You don't need
calibration, markers or measurements.

These 23 phone photos were taken while walking once around a statue:

![Twenty-three photos of a bronze statue of a woman carrying a child, taken from all sides while walking around it](https://raw.githubusercontent.com/d4per/pixelmap/main/images/docs/landala.webp)

The mesh fused from them, shown untextured in MeshLab from three sides:

![The reconstructed 3D mesh of the statue, seen from three different angles](https://raw.githubusercontent.com/d4per/pixelmap/main/images/docs/landala3D.webp)

## Install

```console
pip install pixelmap-multiview-python            # the library; NumPy is the only dependency
pip install "pixelmap-multiview-python[images]"  # plus Pillow, for load_photos()
```

The distribution is `pixelmap-multiview-python`, and the import name is
`pixelmap_multiview`. Wheels are published for Linux (x86-64, aarch64, musl), macOS (Apple
silicon and Intel) and Windows (x86-64), for CPython 3.9 and newer. You only need a Rust
toolchain to build from source.

## Quick start

```python
import pixelmap_multiview as pmv

photos, focal = pmv.load_photos(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
model = pmv.reconstruct(photos, quality="medium", focal_35mm=focal)

print(model)  # <pixelmap_multiview.Model 23314 vertices, 45708 triangles, 4 of 4 photos>
model.save("out/statue.obj")  # + statue.mtl + statue_texture.png, for MeshLab or Blender
model.save("out/statue.html")  # open it in a browser and turn the model around
```

`load_photos` turns each photo upright according to its EXIF orientation and scales it
to 1200 px on the long side. It also returns the 35 mm-equivalent focal length from EXIF
when every photo records the same one. Photos from anywhere else work too: pass a list
of uint8 arrays `(H, W, 3)`, `(H, W, 4)` or `(H, W)`, or Pillow images, all the same size.

## What you get

| Attribute | Contents |
|---|---|
| `model.vertices`, `model.normals` | `(V, 3)` float64 |
| `model.triangles` | `(T, 3)` uint32, counter-clockwise seen from the normal's side |
| `model.vertex_colours` | `(V, 3)` uint8 |
| `model.texture.atlas` | `(H, W, 4)` uint8, with each part of the surface taken from the photo that sees it best |
| `model.texture.texcoords`, `.face_texcoords`, `.face_views` | the atlas mapping, OBJ convention |
| `model.intrinsics` | `fx, fy, cx, cy`, `.matrix`, and whether the focal length was provided, from EXIF, estimated or refined |
| `model.cameras` | a `Pose` per photo (`rotation`, `translation`, `centre`, `matrix`), or `None` for a photo left out |
| `model.pairs` | what two-view geometry made of each pair: coverage, relative pose, inlier ratio, and why it was rejected |

All of these are in one frame. The first camera of the best pair sits at the origin
looking along +z with +y down the image, and the distance between that pair's cameras is
the unit of length. **Nothing is metric.**

## Saving

`model.save(path)` picks the format from the suffix:

| Suffix | Writes | Opens in |
|---|---|---|
| `.obj` | textured Wavefront OBJ, plus `<stem>.mtl` and `<stem>_texture.png` | MeshLab, Blender, CloudCompare |
| `.x3d` | textured X3D, plus `<stem>_texture.png` | MeshLab, X3D viewers |
| `.glb` | binary glTF 2.0, one file with the texture embedded | Blender, Windows 3D Viewer, three.js, most game engines |
| `.html` | one web page with the textured model embedded, shown in 3D | any browser |
| `.ply` | ASCII PLY with a colour per vertex | viewers that do not load textures |

It returns every file it wrote. The `.html` page is titled after the file's stem unless you
pass `title="..."`. The model is embedded in the page, but the viewer, Google's
`<model-viewer>`, is fetched from cdn.jsdelivr.net, so opening the page needs a network
connection. Every format is turned half a revolution about x so that viewers show the
model upright; the arrays on `model` are not turned.

## Following along and stopping

A run takes minutes of work. `on_event` hears about all of it, as data rather than prose:

```python
def on_event(event):
    if event.progress is not None:  # None for log lines and dropped photos
        print(f"{event.progress:5.1%}  {event.message}")
    if isinstance(event, pmv.PairMapped):
        event.points  # (rows, cols, 2): the dense correspondence for this pair, NaN where unmapped


model = pmv.reconstruct(photos, on_event=on_event)
```

The events are `Started`, `StageProgress`, `PairProgress`, `PairMapped`, `PairRejected`,
`ViewDropped` and `Log`. Each one carries `stage`, `progress` and `message`, along with its
own fields. If `on_event` raises an exception, the run stops at the next checkpoint and
the exception propagates. Ctrl-C stops a run the same way.

To keep your own thread free, run the reconstruction as a `Job`:

```python
job = pmv.Job(photos, quality="medium")
for event in job:  # each event as it happens; ends when the run does
    print(f"{job.status.progress:.0%} {event.message}")
model = job.join()
```

For code with its own event loop, `job.next_event(timeout=0)` polls without waiting, and
`job.status` works from any thread. After `job.cancel()`, `join()` raises `CancelledError`,
well within a second.

## When a capture fails

A run that can't produce a correct model raises an exception rather than returning a
plausible-looking wrong one. Each exception names the stage that failed and says what to
change:

```python
try:
    model = pmv.reconstruct(photos)
except pmv.ReconstructionError as err:
    print(err.stage, err.reason)  # tracks too_few_tracks
    # only 338 points could be followed across three or more photos (need 500);
    # the photos share too little of the scene
    print(err)
```

`InvalidInputError` (also a `ValueError`) covers too few photos, photos of different
sizes, and photos that are too small. `ReconstructionError` covers captures that
can't be reconstructed: a camera that only turned, a flat scene, or photos with too little
in common. Match on `err.reason` rather than on the message. The failure's details
are attributes, such as `err.found` and `err.required`.

For a good capture:
- Step sideways between shots rather than turning on the spot.
- Let neighbouring photos overlap generously.
- Keep one camera at one zoom setting throughout.

## Time complexity

Mapping the pairs dominates: N photos take N(N−1)/2 pixelmap runs, one after another, so
the time grows with the square of the number of photos. Four photos make 6 pairs; the 23
photos of the statue make 253. One pair of 4:3 photos takes, by the long side of the
photos:

| Quality | 800 px | 1200 px | 1800 px |
|---------|--------|---------|---------|
| low     | 0.22 s | 0.23 s  | 0.22 s  |
| medium  | 0.98 s | 0.96 s  | 0.97 s  |
| high    | 2.16 s | 2.90 s  | 4.08 s  |

So the statue at `medium` spends about four minutes mapping pairs. These are medians of
three runs of `pixelmap-multiview --time-pair` from `pixelmap_multiview` 0.1.2, on a
six-core Intel Core i7-9750H laptop. Expect other machines to be proportionally faster or
slower.

pixelmap scales both photos down to a fixed working width before matching them: 400 px
for `low`, 800 px for `medium` and 1600 px for `high`. It never scales them up, so at
`low` and `medium` the size of the input barely changes the time. `high` only works at its
full width with photos at least 1600 px across, which is why it slows down as they grow;
at the 1200 px `load_photos` gives by default it works at 1200 px. Larger photos also give
more precise geometry after matching. Runs are deterministic: the same photos, quality and
`seed` give the same model.

## Development

```console
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest pillow ruff numpy
maturin develop --release   # release matters: the tests run whole reconstructions
pytest
```

## License

MIT
