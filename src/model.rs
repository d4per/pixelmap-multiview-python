//! A finished reconstruction, handed to Python as arrays.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pixelmap_multiview::export;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::convert::{array_from_photo, intrinsics, pose, rows, variant_name};

/// The crate's `Model`, without its diagnostics.
///
/// Arrays are built on each access rather than cached: the Python wrapper reads each one
/// once and holds on to it, and caching here would keep two copies alive.
#[pyclass(module = "pixelmap_multiview._pixelmap_multiview", frozen)]
pub struct Model {
    inner: pixelmap_multiview::Model,
}

impl Model {
    pub fn new(mut inner: pixelmap_multiview::Model) -> Self {
        // Every depth map, every sample's fate and every track: usually more memory than
        // the mesh, and this package does not expose them.
        inner.take_diagnostics();
        Model { inner }
    }
}

/// Runs one of the crate's writers into a buffer and hands it over as bytes.
fn written(
    py: Python<'_>,
    write: impl FnOnce(&mut Vec<u8>) -> std::io::Result<()> + Send,
) -> PyResult<Bound<'_, PyBytes>> {
    let buffer = py
        .detach(|| {
            let mut buffer = Vec::new();
            write(&mut buffer).map(|()| buffer)
        })
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyBytes::new(py, &buffer))
}

#[pymethods]
impl Model {
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows(
            py,
            self.inner.mesh.positions.iter().map(|p| [p.x, p.y, p.z]),
        )
    }

    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows(py, self.inner.mesh.normals.iter().map(|n| [n.x, n.y, n.z]))
    }

    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        rows(py, self.inner.mesh.triangles.iter().copied())
    }

    #[getter]
    fn vertex_colours<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        rows(py, self.inner.texture.vertex_colours.iter().copied())
    }

    #[getter]
    fn atlas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        array_from_photo(py, &self.inner.texture.atlas)
    }

    #[getter]
    fn texcoords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        rows(py, self.inner.texture.texcoords.iter().copied())
    }

    #[getter]
    fn face_texcoords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        rows(py, self.inner.texture.face_texcoords.iter().copied())
    }

    #[getter]
    fn face_views<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let views: Vec<u32> = self.inner.texture.face_views.iter().map(|v| v.0).collect();
        Array1::from_vec(views).into_pyarray(py)
    }

    #[getter]
    fn charts(&self) -> usize {
        self.inner.texture.charts
    }

    #[getter]
    fn atlas_scale(&self) -> f64 {
        self.inner.texture.atlas_scale
    }

    #[getter]
    fn intrinsics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        intrinsics(py, &self.inner.intrinsics)
    }

    /// One dict per photo, `None` for a photo that was left out.
    #[getter]
    fn cameras<'py>(&self, py: Python<'py>) -> PyResult<Vec<Option<Bound<'py, PyDict>>>> {
        self.inner
            .cameras
            .iter()
            .map(|camera| camera.as_ref().map(|p| pose(py, p)).transpose())
            .collect()
    }

    #[getter]
    fn views(&self) -> usize {
        self.inner.views
    }

    #[getter]
    fn connected(&self) -> Vec<u32> {
        self.inner.connected.iter().map(|v| v.0).collect()
    }

    #[getter]
    fn pairs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .pairs
            .iter()
            .map(|report| {
                let d = PyDict::new(py);
                d.set_item("pair", (report.pair.a().0, report.pair.b().0))?;
                d.set_item("coverage", report.coverage)?;
                d.set_item(
                    "pose",
                    report.pose.as_ref().map(|p| pose(py, p)).transpose()?,
                )?;
                d.set_item("inlier_ratio", report.inlier_ratio)?;
                d.set_item("median_angle_deg", report.median_angle_deg)?;
                d.set_item("rejection", report.rejection.as_ref().map(variant_name))?;
                d.set_item(
                    "explanation",
                    report.rejection.as_ref().map(|r| r.to_string()),
                )?;
                Ok(d)
            })
            .collect()
    }

    /// Textured Wavefront OBJ referring to `material_library`.
    fn obj<'py>(&self, py: Python<'py>, material_library: &str) -> PyResult<Bound<'py, PyBytes>> {
        let (mesh, texture) = (&self.inner.mesh, &self.inner.texture);
        written(py, |out| {
            export::write_textured_obj(out, mesh, texture, material_library)
        })
    }

    /// The material library for `obj`, referring to `texture_file`.
    fn mtl<'py>(&self, py: Python<'py>, texture_file: &str) -> PyResult<Bound<'py, PyBytes>> {
        written(py, |out| export::write_mtl(out, texture_file))
    }

    /// ASCII PLY with one colour per vertex.
    fn ply<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let (mesh, colours) = (&self.inner.mesh, &self.inner.texture.vertex_colours);
        written(py, |out| export::write_ply_mesh(out, mesh, colours))
    }

    /// Textured X3D referring to `texture_file`.
    fn x3d<'py>(&self, py: Python<'py>, texture_file: &str) -> PyResult<Bound<'py, PyBytes>> {
        let (mesh, texture) = (&self.inner.mesh, &self.inner.texture);
        written(py, |out| {
            export::write_textured_x3d(out, mesh, texture, texture_file)
        })
    }

    /// The texture atlas as PNG.
    fn atlas_png<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let atlas = &self.inner.texture.atlas;
        written(py, |out| {
            let mut encoder = png::Encoder::new(out, atlas.width() as u32, atlas.height() as u32);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(atlas.as_rgba())?;
            writer.finish()?;
            Ok(())
        })
    }
}
