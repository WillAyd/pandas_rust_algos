#!/bin/bash

pushd $1
cargo build --profile $2
cp "target/$3/libpandas_rust_algos.so" "$4/pandas_rust_algos.so"
popd
