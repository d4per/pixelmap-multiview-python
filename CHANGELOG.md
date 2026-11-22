# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.0 - 2026-11-22

### Added

- `neighborhood_radius` and `init_neighborhood_radius` keywords for `reconstruct` and
  `Job`: the radius of the disc pixelmap compares two photos over when it maps each pair,
  in every solve or in the initial solve alone. Each is an int from 1 to the new
  `MAX_NEIGHBORHOOD_RADIUS` (51), or `None`, the default, for pixelmap's own radius of 5.
  A radius out of range raises `ValueError` when the call is made, not partway through
  the run.
- `Quality.ULTRA` (`"ultra"`): `HIGH`, then two more steps at the photos' own width, up
  to 3200 px. On photos no wider than 1600 px it is the same as `HIGH`.

### Changed

- Built against `pixelmap_multiview` 0.2.0 and pixelmap 0.4.1.
- **`medium` and `high` no longer scale photos up.** They used to scale photos narrower
  than their working width (800 and 1600 px) up to it. They now work at the photos' own
  width instead, which pixelmap found more accurate and is faster, but it gives a sparser
  mesh. With `load_photos`' default of 1200 px, portrait photos are 900 px wide: on three
  such photos `high` gave 81,526 vertices where 0.1.2 gave 322,885. Pass `load_photos` a
  larger `max_size`, or `None`, for the density back. Photos at least as wide as the
  working width give the same model as 0.1.2, bit for bit, and `low` always does.
- The README's timings are re-measured against 0.2.0 and have an `ultra` row and a
  2400×3200 column.

## 0.1.2 - 2026-11-17

### Changed

- Built against `pixelmap_multiview` 0.1.2 and pixelmap 0.3.1. Mapping each pair of
  photos takes about 40% fewer region comparisons, so `reconstruct` is faster: on
  4128x3096 photos one pair at `high` went from 5.9 s to 4.3 s.
- **The same photos, quality and `seed` no longer give the model 0.1.1 gave.** Runs are
  still deterministic, but the solver reaches a different, equally good correspondence.

### Fixed

- Gaps in the depth maps and the mesh. Some pairs' correspondence had straight unmapped
  bands, one grid cell wide, running through it; pixelmap 0.3.1 no longer leaves them.

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
