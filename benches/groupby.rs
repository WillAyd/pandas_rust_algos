#![feature(test)]
extern crate test;

use numpy::ndarray::{Array1, Array2};
use pandas_rust_algos::groupby::group_sum;
use test::Bencher;

#[bench]
fn bench_group_sum(b: &mut Bencher) {
    let n = 10_000_000;
    let n_labels = 200;
    let mut out = Array2::<i64>::default((n, 1));
    let values = Array2::<i64>::ones((n, 1));
    let mut counts = Array1::<i64>::zeros(n_labels);

    let labels_arr = Array1::from_iter(0..n);
    let labels = labels_arr.map(|x| (x % n_labels) as i64);

    let min_count = -1;
    let is_datetimelike = false;

    b.iter(|| {
        group_sum(
            out.view_mut(),
            counts.view_mut(),
            values.view(),
            labels.view(),
            None, // mask
            None, // result_mask
            min_count,
            is_datetimelike,
        )
    });
}
