# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.1 - 2026-10-04

### Added

- `Model.save` writes two more formats: `.glb`, binary glTF 2.0 with the texture atlas
  embedded, and `.html`, one self-contained page that shows the textured model in 3D in
  a browser. A new `title` keyword names the page.

### Changed

- Built against `pixelmap_multiview` 0.1.1, which provides the GLB and HTML writers.

## 0.1.0 - 2026-09-26

First release, built against `pixelmap_multiview` 0.1.0.

### Added

- `reconstruct`, which goes from photos to a textured mesh in one call, with typed
  progress events through `on_event`. An exception raised there, or Ctrl-C, stops the run.
- `Job`, which runs a reconstruction on a background thread that can be iterated, polled
  and cancelled.
- `Model`, which holds the mesh, vertex colours, texture atlas, intrinsics, a pose per
  photo and a report on every pair as NumPy arrays and dataclasses. `Model.save` writes
  OBJ with MTL and PNG, X3D, or PLY.
- `load_photos`, which reads photos with Pillow, turns them upright, scales them, and
  reads the 35 mm-equivalent focal length from EXIF.
- `InvalidInputError`, `ReconstructionError` and `CancelledError`, which carry the failing
  stage, a stable `reason` and the failure's details as attributes.
