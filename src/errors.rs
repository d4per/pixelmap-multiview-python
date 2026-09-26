//! Translating `pixelmap_multiview::Error` into Python exceptions.

use pixelmap_multiview::{Error, Stage};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::variant_name;

/// Raises the Python exception that corresponds to `err`.
///
/// The classes live in `pixelmap_multiview._errors` rather than being created here: the
/// input errors inherit from both `PixelmapMultiviewError` and `ValueError`, so that
/// `except ValueError` keeps working for callers who do not know this library's hierarchy.
/// Two bases is a plain class statement in Python and a fight with `create_exception!` in
/// Rust.
///
/// Every exception carries `stage` and `reason`, plus the variant's own fields as
/// attributes, so that a caller can act on a failure without parsing its message.
pub fn to_pyerr(py: Python<'_>, err: Error) -> PyErr {
    let message = err.to_string();
    let stage = stage_name(err.stage());
    let reason = variant_name(&err);
    let class = match &err {
        Error::TooFewPhotos { .. }
        | Error::SizeMismatch { .. }
        | Error::PhotoTooSmall { .. }
        | Error::InvalidIntrinsics
        | Error::GraphMismatch { .. } => "InvalidInputError",
        Error::Cancelled { .. } => "CancelledError",
        _ => "ReconstructionError",
    };
    build(py, class, &message, |kwargs| {
        kwargs.set_item("stage", stage)?;
        kwargs.set_item("reason", &reason)?;
        fields(&err, kwargs)
    })
}

/// A `pixelmap::Error` from turning an array into a photo. The Python layer has already
/// checked the shape, so reaching this means the buffer itself was unusable.
pub fn photo_error(err: pixelmap_multiview::pixelmap::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// The stage's short name, as `STAGES` lists them on the Python side.
pub fn stage_name(stage: Stage) -> String {
    variant_name(&stage)
}

/// The variant's fields, as keyword arguments to the exception.
fn fields(err: &Error, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
    let views = |ids: &[pixelmap_multiview::ViewId]| ids.iter().map(|v| v.0).collect::<Vec<_>>();
    match err {
        Error::TooFewPhotos { found, minimum } => {
            kwargs.set_item("found", found)?;
            kwargs.set_item("minimum", minimum)
        }
        Error::SizeMismatch {
            view,
            expected,
            found,
        } => {
            kwargs.set_item("view", view.0)?;
            kwargs.set_item("expected", expected)?;
            kwargs.set_item("found", found)
        }
        Error::PhotoTooSmall {
            view,
            dimensions,
            minimum,
        } => {
            kwargs.set_item("view", view.0)?;
            kwargs.set_item("dimensions", dimensions)?;
            kwargs.set_item("minimum", minimum)
        }
        Error::TooFewTracks { found, required } => {
            kwargs.set_item("found", found)?;
            kwargs.set_item("required", required)
        }
        Error::NoUsablePair { reasons } => kwargs.set_item(
            "pairs",
            reasons
                .iter()
                .map(|(pair, why)| ((pair.a().0, pair.b().0), why.to_string()))
                .collect::<Vec<_>>(),
        ),
        Error::RegistrationFailed {
            registered,
            minimum,
            left_out,
        } => {
            kwargs.set_item("registered", views(registered))?;
            kwargs.set_item("minimum", minimum)?;
            kwargs.set_item(
                "left_out",
                left_out
                    .iter()
                    .map(|(view, why)| (view.0, why.to_string()))
                    .collect::<Vec<_>>(),
            )
        }
        Error::BundleAdjustment {
            initial_median_px,
            median_px,
            required_px,
        } => {
            kwargs.set_item("initial_median_px", initial_median_px)?;
            kwargs.set_item("median_px", median_px)?;
            kwargs.set_item("required_px", required_px)
        }
        Error::InsufficientDepth {
            valid_fraction,
            required,
        } => {
            kwargs.set_item("valid_fraction", valid_fraction)?;
            kwargs.set_item("required", required)
        }
        Error::FragmentedMesh {
            largest_component,
            required,
        } => {
            kwargs.set_item("largest_component", largest_component)?;
            kwargs.set_item("required", required)
        }
        Error::Correspondence { pair, .. } => kwargs.set_item("pair", (pair.a().0, pair.b().0)),
        Error::DisconnectedViews {
            connected,
            minimum,
            min_coverage,
        } => {
            kwargs.set_item("connected", views(connected))?;
            kwargs.set_item("minimum", minimum)?;
            kwargs.set_item("min_coverage", min_coverage)
        }
        // `InvalidIntrinsics`, `EmptyMesh` and `Cancelled` carry nothing beyond the stage,
        // and a variant added in a later crate version still raises, with its message.
        _ => Ok(()),
    }
}

fn build(
    py: Python<'_>,
    class_name: &str,
    message: &str,
    fill: impl FnOnce(&Bound<'_, PyDict>) -> PyResult<()>,
) -> PyErr {
    let construct = || -> PyResult<PyErr> {
        let class = py
            .import("pixelmap_multiview._errors")?
            .getattr(class_name)?;
        let kwargs = PyDict::new(py);
        fill(&kwargs)?;
        Ok(PyErr::from_value(class.call((message,), Some(&kwargs))?))
    };
    // Failing to build the exception is itself an exception worth surfacing; it means the
    // Python half of the package is missing or out of step with this module.
    construct().unwrap_or_else(|failure| failure)
}
