use numpy::ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::PyResult;
use std::fmt::Debug;

#[pymodule]
fn pandas_rust_algos(_py: Python, m: &PyModule) -> PyResult<()> {
    #[derive(FromPyObject)]
    enum TakeType<'py> {
        U8(&'py PyArray1<u8>),
        I8(&'py PyArray1<i8>),
        I16(&'py PyArray1<i16>),
        I32(&'py PyArray1<i32>),
        I64(&'py PyArray1<i64>),
        F32(&'py PyArray1<f32>),
        F64(&'py PyArray1<f64>),
    }

    fn take_1d<InT, OutT>(
        values: ArrayView1<InT>,
        indexer: ArrayView1<i64>,
        mut out: ArrayViewMut1<OutT>,
        fill_value: OutT,
    ) where
        InT: std::marker::Copy,
        OutT: TryFrom<InT> + std::marker::Copy,
        <OutT as TryFrom<InT>>::Error: Debug,
    {
        for (i, idx) in indexer.iter().enumerate() {
            if *idx == -1 {
                out[i] = fill_value;
            } else {
                let uidx = usize::try_from(*idx).unwrap();
                out[i] = OutT::try_from(values[uidx]).unwrap();
            }
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "take_1d")]
    fn take_1d_py<'py>(values: TakeType<'py>, indexer: PyReadonlyArray1<i64>, out: TakeType<'py>) {
        match (values, out) {
            (TakeType::U8(values), TakeType::U8(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::I8(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::I32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::I16(values), TakeType::I32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I16(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I16(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::I32(values), TakeType::I32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I32(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I32(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::I64(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            // TODO: the trait `From<i64>` is not implemented for `f64`
            /*
                (TakeType::I64(values), TakeType::F64(out)) => take_1d(
                    values.readonly().as_array(),
                    indexer.as_array(),
                    out.readwrite().as_array_mut(),
                    0.0,
            ),
             */
            (TakeType::F32(values), TakeType::F32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::F32(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::F64(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (_, _) => panic!("Types not supported"),
        }
    }

    fn take_2d(values: ArrayView2<i64>, indexer: ArrayView1<i64>, mut out: ArrayViewMut2<i64>) {
        let nrows = values.raw_dim()[0];
        let ncols = indexer.raw_dim()[0];

        for i in 0..nrows {
            for j in 0..ncols {
                unsafe {
                    let idx = *indexer.uget(j);
                    *out.uget_mut((i, j)) = *values.uget((i, idx as usize));
                }
            }
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "take_2d")]
    fn take_2d_py<'py>(
        values: PyReadonlyArray2<i64>,
        indexer: PyReadonlyArray1<i64>,
        mut out: PyReadwriteArray2<i64>,
    ) {
        take_2d(values.as_array(), indexer.as_array(), out.as_array_mut())
    }
    Ok(())
}
