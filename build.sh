#!/bin/bash
SOURCE_DIR="/home/jdomagala/Work/SmollNet"
BUILD_DIR="${SOURCE_DIR}/build"

mkdir -d ${BUILD_DIR}
cd ${BUILD_DIR} || exit 0

CUDA_DIR="/usr/local/cuda-12.8"

cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCUDA_ROOT=${CUDA_DIR} \
    --fresh \
    ..

make -j 24
