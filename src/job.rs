//! A reconstruction on a worker thread, for Python callers with their own loop to turn.

use std::sync::mpsc::RecvTimeoutError;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use pixelmap_multiview::Cancel;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::to_pyerr;
use crate::events;
use crate::model::Model;

/// How long a wait holds the job's lock, and goes without checking for Ctrl-C.
///
/// Waiting in slices rather than all at once is what keeps `status` and `cancel`
/// responsive while another thread sits in `next_event`, and Ctrl-C prompt in a
/// single-threaded one.
const SLICE: Duration = Duration::from_millis(50);

/// Wraps the crate's `Job`, which already folds events into a status and owns the cancel
/// token. `None` once `join` has taken it.
#[pyclass(module = "pixelmap_multiview._pixelmap_multiview", frozen)]
pub struct Job {
    job: Mutex<Option<pixelmap_multiview::Job>>,
    // Held apart from the job so that cancelling never waits on the lock.
    cancel: Cancel,
}

impl Job {
    pub fn new(job: pixelmap_multiview::Job) -> Self {
        let cancel = job.canceller();
        Job {
            job: Mutex::new(Some(job)),
            cancel,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<pixelmap_multiview::Job>> {
        // A panic while holding the lock leaves nothing half-updated worth refusing over.
        self.job
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

enum Next {
    Event(pixelmap_multiview::Event),
    Waiting,
    Finished,
}

#[pymethods]
impl Job {
    /// The next event as `(kind, fields)`, or `None` if `timeout` seconds pass first.
    /// Raises `StopIteration` once the run is over and every event has been read.
    #[pyo3(signature = (timeout = None))]
    fn next_event<'py>(
        &self,
        py: Python<'py>,
        timeout: Option<f64>,
    ) -> PyResult<Option<(String, Bound<'py, PyDict>)>> {
        let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
        loop {
            let slice = match deadline {
                Some(deadline) => deadline
                    .saturating_duration_since(Instant::now())
                    .min(SLICE),
                None => SLICE,
            };
            let next = py.detach(|| match self.lock().as_ref() {
                None => Next::Finished,
                Some(job) => match job.events().recv_timeout(slice) {
                    Ok(event) => Next::Event(event),
                    Err(RecvTimeoutError::Timeout) => Next::Waiting,
                    Err(RecvTimeoutError::Disconnected) => Next::Finished,
                },
            });
            match next {
                Next::Event(event) => return events::to_python(py, &event).map(Some),
                Next::Finished => {
                    return Err(pyo3::exceptions::PyStopIteration::new_err(()));
                }
                Next::Waiting => {
                    py.check_signals()?;
                    if deadline.is_some_and(|d| Instant::now() >= d) {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Where the run has got to.
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let status = py.detach(|| self.lock().as_ref().map(|job| job.status()));
        match status {
            Some(status) => events::status(py, &status),
            None => Err(PyRuntimeError::new_err("the job has already been joined")),
        }
    }

    /// Asks the run to stop. Returns at once.
    fn cancel(&self) {
        self.cancel.cancel();
    }

    #[getter]
    fn cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Waits for the run to finish and gives its model.
    ///
    /// Drains any events nobody read: the worker drops its end of the channel when it is
    /// done, which is the only way to learn that without blocking in `join` itself, where
    /// Ctrl-C could not reach.
    fn join(&self, py: Python<'_>) -> PyResult<Model> {
        loop {
            let finished = py.detach(|| match self.lock().as_ref() {
                None => true,
                Some(job) => loop {
                    match job.events().recv_timeout(SLICE) {
                        Ok(_) => continue,
                        Err(RecvTimeoutError::Timeout) => break false,
                        Err(RecvTimeoutError::Disconnected) => break true,
                    }
                },
            });
            if finished {
                break;
            }
            if let Err(err) = py.check_signals() {
                // Leave the worker to wind down rather than grind on for nobody.
                self.cancel.cancel();
                return Err(err);
            }
        }
        let job = self
            .lock()
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("the job has already been joined"))?;
        match py.detach(|| job.join()) {
            Ok(model) => Ok(Model::new(model)),
            Err(err) => Err(to_pyerr(py, err)),
        }
    }
}
