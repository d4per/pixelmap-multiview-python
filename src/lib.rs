//! Native half of the `pixelmap_multiview` Python package.
//!
//! Everything user-facing — argument normalisation, docstrings, the event and exception
//! classes — lives in the Python layer next door. This module does the FFI and nothing
//! else.

mod convert;
mod errors;
mod events;
mod job;
mod model;

use std::str::FromStr;
use std::sync::Arc;

use numpy::PyReadonlyArray3;
use pixelmap_multiview::pixelmap::{Photo, Quality};
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{Flow, Focal, Options, Stage, ViewId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{array_from_photo, photo_from_array};
use crate::errors::{photo_error, stage_name, to_pyerr};

fn photos(arrays: Vec<PyReadonlyArray3<'_, u8>>) -> PyResult<Vec<Arc<Photo>>> {
    arrays
        .iter()
        .map(|array| photo_from_array(array).map(Arc::new).map_err(photo_error))
        .collect()
}

/// The crate's `Options`, from arguments the Python layer has already checked.
///
/// `focal_kind` is `"unknown"`, `"35mm"` or `"pixels"`, with `focal_value` in those units.
fn options(
    quality: &str,
    focal_kind: &str,
    focal_value: f64,
    seed: Option<u64>,
    refine_focal: bool,
    max_texture_size: usize,
) -> PyResult<Options> {
    let quality =
        Quality::from_str(quality).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let focal = match focal_kind {
        "unknown" => Focal::Unknown,
        "35mm" => Focal::Equivalent35mm(focal_value),
        "pixels" => Focal::Pixels(focal_value),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown focal kind {other:?}"
            )))
        }
    };
    Ok(Options::new()
        .quality(quality)
        .focal(focal)
        .seed(seed)
        .refine_focal(refine_focal)
        .max_texture_size(max_texture_size))
}

/// Reconstructs a model from photos, blocking until it is done.
///
/// `on_event(kind, fields)` is called at every checkpoint. An exception it raises, or a
/// Ctrl-C arriving meanwhile, stops the run at that checkpoint and is raised once it has
/// stopped. Signals are checked at every event, not only when there is a callback, so
/// Ctrl-C works without one.
#[pyfunction]
#[pyo3(signature = (photos, quality, focal_kind, focal_value, seed, refine_focal, max_texture_size, on_event))]
#[allow(clippy::too_many_arguments)]
fn run(
    py: Python<'_>,
    photos: Vec<PyReadonlyArray3<'_, u8>>,
    quality: &str,
    focal_kind: &str,
    focal_value: f64,
    seed: Option<u64>,
    refine_focal: bool,
    max_texture_size: usize,
    on_event: Option<Py<PyAny>>,
) -> PyResult<model::Model> {
    let options = options(
        quality,
        focal_kind,
        focal_value,
        seed,
        refine_focal,
        max_texture_size,
    )?;
    let photos = self::photos(photos)?;

    let mut interrupted: Option<PyErr> = None;
    let result = py.detach(|| {
        pixelmap_multiview::run(&photos, &options, &mut |event| {
            Python::attach(|py| {
                let outcome = py.check_signals().and_then(|()| match on_event.as_ref() {
                    Some(callback) => {
                        let (kind, fields) = events::to_python(py, &event)?;
                        callback.call1(py, (kind, fields)).map(|_| ())
                    }
                    None => Ok(()),
                });
                match outcome {
                    Ok(()) => Flow::Continue(()),
                    Err(err) => {
                        interrupted = Some(err);
                        Flow::Break(())
                    }
                }
            })
        })
    });

    // The callback's own exception, not the `Cancelled` the crate turned it into.
    if let Some(err) = interrupted {
        return Err(err);
    }
    result
        .map(model::Model::new)
        .map_err(|err| to_pyerr(py, err))
}

/// Starts a reconstruction on a worker thread.
///
/// Nothing is validated here, as in the crate: anything wrong with the photos comes back
/// from `join` as the same exception `run` would have raised.
#[pyfunction]
#[pyo3(signature = (photos, quality, focal_kind, focal_value, seed, refine_focal, max_texture_size))]
fn start(
    photos: Vec<PyReadonlyArray3<'_, u8>>,
    quality: &str,
    focal_kind: &str,
    focal_value: f64,
    seed: Option<u64>,
    refine_focal: bool,
    max_texture_size: usize,
) -> PyResult<job::Job> {
    let options = options(
        quality,
        focal_kind,
        focal_value,
        seed,
        refine_focal,
        max_texture_size,
    )?;
    Ok(job::Job::new(pixelmap_multiview::Job::start(
        self::photos(photos)?,
        options,
    )))
}

/// Photos of one of the crate's synthetic scenes, from cameras on an arc around it.
///
/// For this package's tests only: the scenes are the crate's test fixtures, hidden from
/// its documentation and liable to change. Not part of the public API.
#[pyfunction]
fn _synthetic_orbit<'py>(
    py: Python<'py>,
    scene: &str,
    views: usize,
    spread_deg: f64,
    width: usize,
    height: usize,
) -> PyResult<Vec<Bound<'py, numpy::PyArray3<u8>>>> {
    let scene = Scene::named(scene)
        .ok_or_else(|| PyValueError::new_err(format!("no synthetic scene {scene:?}")))?;
    let set = SyntheticSet::orbit(scene, views, spread_deg, width, height);
    let rendered: Vec<Photo> = py.detach(|| {
        (0..set.views())
            .map(|v| set.render(ViewId(v as u32)))
            .collect()
    });
    Ok(rendered.iter().map(|p| array_from_photo(py, p)).collect())
}

#[pymodule]
fn _pixelmap_multiview(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::Model>()?;
    m.add_class::<job::Job>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(_synthetic_orbit, m)?)?;
    m.add("MIN_VIEWS", pixelmap_multiview::MIN_VIEWS)?;
    m.add("MIN_PAIR_COVERAGE", pixelmap_multiview::MIN_PAIR_COVERAGE)?;
    m.add(
        "DEFAULT_MAX_TEXTURE_SIZE",
        pixelmap_multiview::DEFAULT_MAX_TEXTURE_SIZE,
    )?;
    m.add("MIN_DIMENSION", pixelmap_multiview::pixelmap::MIN_DIMENSION)?;
    m.add(
        "STAGES",
        Stage::ALL
            .iter()
            .map(|&s| (stage_name(s), s.name()))
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
