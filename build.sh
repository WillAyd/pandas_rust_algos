#!/bin/bash

pushd $1
cargo build $2
cp target/release/libpandas_rust_algos.so pandas_rust_algos.so
popd
