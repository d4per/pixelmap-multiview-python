//! Handing `pixelmap_multiview::Event`s to Python.
//!
//! Each event crosses as `(kind, fields)`, and `pixelmap_multiview._events` turns that into
//! a dataclass. `progress` and `message` are computed here, by the crate, so the Python
//! side never re-implements the stage weights or the wording.

use pixelmap_multiview::{Event, PairId, Status};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{pair_points, variant_name};
use crate::errors::stage_name;

fn pair(pair: &PairId) -> (u32, u32) {
    (pair.a().0, pair.b().0)
}

/// `event` as `(kind, fields)`.
pub fn to_python<'py>(py: Python<'py>, event: &Event) -> PyResult<(String, Bound<'py, PyDict>)> {
    let d = PyDict::new(py);
    d.set_item("stage", stage_name(event.stage()))?;
    d.set_item("progress", event.progress())?;
    d.set_item("message", event.message().as_ref())?;
    let kind = match event {
        Event::Started { views, pairs } => {
            d.set_item("views", views)?;
            d.set_item("pairs", pairs)?;
            "started"
        }
        Event::Stage { fraction, .. } => {
            d.set_item("fraction", fraction)?;
            "stage"
        }
        Event::PairProgress {
            pair: p,
            index,
            of,
            step,
            steps,
        } => {
            d.set_item("pair", pair(p))?;
            d.set_item("index", index)?;
            d.set_item("of", of)?;
            d.set_item("step", step)?;
            d.set_item("steps", steps)?;
            "pair_progress"
        }
        Event::PairMapped {
            pair: p,
            index,
            of,
            map,
        } => {
            d.set_item("pair", pair(p))?;
            d.set_item("index", index)?;
            d.set_item("of", of)?;
            d.set_item("points", pair_points(py, map))?;
            d.set_item("cell_size", map.cell_size)?;
            d.set_item("coverage", map.coverage)?;
            "pair_mapped"
        }
        Event::PairRejected {
            pair: p,
            index,
            of,
            reason,
        } => {
            d.set_item("pair", pair(p))?;
            d.set_item("index", index)?;
            d.set_item("of", of)?;
            d.set_item("reason", variant_name(reason))?;
            d.set_item("explanation", reason.to_string())?;
            "pair_rejected"
        }
        Event::ViewDropped { view, reason, .. } => {
            d.set_item("view", view.0)?;
            d.set_item("reason", variant_name(reason))?;
            d.set_item("explanation", reason.to_string())?;
            "view_dropped"
        }
        Event::Log { level, .. } => {
            d.set_item("level", level.to_string())?;
            "log"
        }
        // A variant from a later crate version still reaches the caller, as a plain event
        // with its stage, progress and message.
        _ => "other",
    };
    Ok((kind.to_owned(), d))
}

pub fn status<'py>(py: Python<'py>, status: &Status) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("stage", stage_name(status.stage))?;
    d.set_item("progress", status.progress)?;
    d.set_item("pairs_mapped", status.pairs_mapped)?;
    d.set_item("pairs_total", status.pairs_total)?;
    d.set_item("message", &status.message)?;
    Ok(d)
}
