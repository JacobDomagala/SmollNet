#!/usr/bin/env bash
set -e

SOURCE=/home/jdomagala/Work/SmollNet
BUILD=$SOURCE/build

CUDA=/usr/local/cuda-12.8
CLANG=/usr/local/bin

mkdir -p "$BUILD"

cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_CXX_COMPILER="$CLANG/clang++" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="$BUILD/smollnet" \
  --fresh

cmake --build "$BUILD" --target install

SOURCE=/home/jdomagala/Work/SmollNet/example
cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_CXX_COMPILER="$CLANG/clang++" \
  -DSmollNet_ROOT=${BUILD}/smollnet \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  --fresh

cmake --build "$BUILD"
