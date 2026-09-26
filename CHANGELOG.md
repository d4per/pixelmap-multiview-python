# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - unreleased

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
