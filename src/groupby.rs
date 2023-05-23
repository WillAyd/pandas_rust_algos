use crate::algos::groupsort_indexer;
use numpy::ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2};

pub fn group_median_float64(
    out: ArrayViewMut2<f64>,
    counts: ArrayView1<i64>,
    values: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    min_count: isize,
    mask: ArrayView2<u8>,
    result_mask: ArrayView2<u8>,
) {
    if min_count == -1 {
        panic!("'min_count' only used in sum and prod");
    }

    let ngroups = counts.len();
    let dim = values.raw_dim();
    let N = dim[0];
    let K = dim[1];

    let (indexer, _counts) = groupsort_indexer(labels, ngroups);
    // counts[:] = _counts[1:]

    let data = Array2::<f64>::default((N, K));
}
