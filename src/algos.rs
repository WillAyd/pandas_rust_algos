use numpy::ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2, Axis};

pub fn take_2d<T>(values: ArrayView2<T>, indexer: ArrayView1<i64>, mut out: ArrayViewMut2<T>)
where
    T: Copy,
{
    let ncols = indexer.raw_dim()[0];

    for (i, val_row) in values.axis_iter(Axis(0)).enumerate() {
        for j in 0..ncols {
            unsafe {
                let idx = *indexer.uget(j);
                *out.uget_mut((i, j)) = val_row[idx as usize];
            }
        }
    }
}

pub fn groupsort_indexer(index: ArrayView1<i64>, ngroups: usize) -> (Array1<i64>, Array1<i64>) {
    let mut counts = Array1::<i64>::zeros(ngroups + 1);
    let n = index.len();
    let mut indexer = Array1::<i64>::zeros(n);
    let mut where_ = Array1::<i64>::zeros(ngroups + 1);

    for i in 0..n {
        let idx = index[i];
        counts[(idx + 1) as usize] += 1;
    }

    for i in 1..ngroups + 1 {
        where_[i] = where_[i - 1] + counts[i - 1];
    }

    for i in 0..n {
        let label = index[i] + 1;
        indexer[label as usize] = i as i64;
        where_[label as usize] += 1;
    }

    (indexer, counts)
}
