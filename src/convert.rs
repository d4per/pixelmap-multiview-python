//! Moving data between NumPy arrays and the crate's types.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray3};
use pixelmap_multiview::nalgebra::{Matrix3, Vector3};
use pixelmap_multiview::pixelmap::Photo;
use pixelmap_multiview::{Intrinsics, PairMap, Pose};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Builds a `Photo` from an `(H, W, 3)` or `(H, W, 4)` uint8 array.
///
/// The Python layer has already normalised dtype, rank and channel count, so the only
/// thing left to handle here is a stride pattern NumPy would not flatten for us.
pub fn photo_from_array(
    array: &PyReadonlyArray3<'_, u8>,
) -> Result<Photo, pixelmap_multiview::pixelmap::Error> {
    let view = array.as_array();
    let shape = view.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);

    // `as_slice` is None for a non-contiguous view. The Python layer calls
    // `ascontiguousarray`, so this is belt and braces rather than the expected path —
    // but copying is still better than panicking if that ever stops being true.
    let copied;
    let data: &[u8] = match view.as_slice() {
        Some(slice) => slice,
        None => {
            copied = view.iter().copied().collect::<Vec<u8>>();
            &copied
        }
    };

    if channels == 4 {
        Photo::from_rgba(width, height, data.to_vec())
    } else {
        Photo::from_rgb(width, height, data)
    }
}

/// Hands a photo's pixels back to Python as an `(H, W, 4)` uint8 array.
pub fn array_from_photo<'py>(py: Python<'py>, photo: &Photo) -> Bound<'py, PyArray3<u8>> {
    let (width, height) = (photo.width(), photo.height());
    Array3::from_shape_vec((height, width, 4), photo.as_rgba().to_vec())
        .expect("a Photo's buffer is always width * height * 4 bytes")
        .into_pyarray(py)
}

/// `rows` of `N` values as an `(len, N)` array.
pub fn rows<T: numpy::Element + Copy, const N: usize>(
    py: Python<'_>,
    rows: impl ExactSizeIterator<Item = [T; N]>,
) -> Bound<'_, PyArray2<T>> {
    let len = rows.len();
    let flat: Vec<T> = rows.flat_map(|row| row.into_iter()).collect();
    Array2::from_shape_vec((len, N), flat)
        .expect("every row has N values")
        .into_pyarray(py)
}

pub fn matrix3<'py>(py: Python<'py>, m: &Matrix3<f64>) -> Bound<'py, PyArray2<f64>> {
    rows(py, (0..3).map(|r| [m[(r, 0)], m[(r, 1)], m[(r, 2)]]))
}

pub fn vector3<'py>(py: Python<'py>, v: &Vector3<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(vec![v.x, v.y, v.z]).into_pyarray(py)
}

/// A pose as a dict of `rotation` `(3, 3)` and `translation` `(3,)`, world to camera.
pub fn pose<'py>(py: Python<'py>, pose: &Pose) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("rotation", matrix3(py, pose.rotation.matrix()))?;
    dict.set_item("translation", vector3(py, &pose.translation))?;
    Ok(dict)
}

pub fn intrinsics<'py>(py: Python<'py>, k: &Intrinsics) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("fx", k.fx)?;
    dict.set_item("fy", k.fy)?;
    dict.set_item("cx", k.cx)?;
    dict.set_item("cy", k.cy)?;
    dict.set_item("source", variant_name(&k.source))?;
    Ok(dict)
}

/// A pair map's grid as `(rows, columns, 2)` float32, `NaN` where a cell is unmapped.
pub fn pair_points<'py>(py: Python<'py>, map: &PairMap) -> Bound<'py, PyArray3<f32>> {
    let mut points = map.points.clone();
    // The crate promises `columns * rows * 2` values; pad rather than panic if a future
    // version ever hands over a short buffer.
    points.resize(map.rows * map.columns * 2, f32::NAN);
    Array3::from_shape_vec((map.rows, map.columns, 2), points)
        .expect("resized to rows * columns * 2")
        .into_pyarray(py)
}

/// The variant name of a fieldless or struct-like enum, in snake_case: `TooFewMatches {
/// .. }` becomes `too_few_matches`.
///
/// Read from `Debug` because the crate's enums are `#[non_exhaustive]`: a variant added
/// in a later version still gets a sensible name here without this module knowing it.
pub fn variant_name(value: &impl std::fmt::Debug) -> String {
    let debug = format!("{value:?}");
    let name = debug
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or_default();
    let mut snake = String::with_capacity(name.len() + 4);
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                snake.push('_');
            }
            snake.extend(c.to_lowercase());
        } else {
            snake.push(c);
        }
    }
    snake
}
