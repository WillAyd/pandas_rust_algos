use numpy::PyArray2;
use pyo3::FromPyObject;

#[derive(FromPyObject)]
pub enum NumericArray2<'py> {
    I8(&'py PyArray2<i8>),
    I16(&'py PyArray2<i16>),
    I32(&'py PyArray2<i32>),
    I64(&'py PyArray2<i64>),

    U8(&'py PyArray2<u8>),
    U16(&'py PyArray2<u16>),
    U32(&'py PyArray2<u32>),
    U64(&'py PyArray2<u64>),

    F32(&'py PyArray2<f32>),
    F64(&'py PyArray2<f64>),
}