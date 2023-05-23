use crate::algos::{groupsort_indexer, take_2d_axis1};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2};

unsafe fn calc_median_linear(a: *const f64, n: i64, na_count: i64) -> f64 {
    let mut result;

    if (n % 2) > 0 {
        result = 1.;
    } else {
        result = 0.;
    }

    result
}

unsafe fn median_linear_mask(a: *const f64, n: i64, mask: *const u8) -> f64 {
    let mut na_count = 0;

    if n == 0 {
        return f64::NAN;
    }

    for i in 0..n {
        if *mask.add(i as usize) == 1 {
            na_count += 1;
        }
    }

    // TOOD: implement for NA
    /*
        if na_count > 0 {
            if na_count == n {
                return f64::NAN;
            }

        ... raw allocation in pandas
    }
     */

    let result = calc_median_linear(a, n, na_count);

    result
}

pub fn group_median_float64(
    mut out: ArrayViewMut2<f64>,
    counts: ArrayView1<i64>,
    values: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    min_count: isize,
    mask: ArrayView2<u8>,
    mut result_mask: ArrayViewMut2<u8>,
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

    let mut data = Array2::<f64>::default((N, K));
    let mut ptr = data.as_ptr();

    take_2d_axis1(values.t(), indexer.view(), data.view_mut());

    // if_uses_mask: block in groupby; here assume we always do
    let mut data_mask = Array2::<u8>::default((K, N));
    let mut ptr_mask = data_mask.as_ptr();

    // TODO: cython has fill_value=1 here as well
    take_2d_axis1(mask.t(), indexer.view(), data_mask.view_mut());

    for i in 0..K {
        unsafe {
            let increment = _counts[0] as usize;
            ptr = ptr.add(increment);
            ptr_mask = ptr_mask.add(increment);

            for j in 0..ngroups {
                let size = _counts[j + 1];
                // result = median_linear_mask(ptr, size, ptr_mask)
                out[(j, i)] = 0.;

                // todo: specialize for f64 somehow
                // if result.is_nan()
                // result_mask[(j, i)] = 1
                ptr = ptr.add(size as usize);
                ptr_mask = ptr_mask.add(size as usize);
            }
        }
    }
}
