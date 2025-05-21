#!/usr/bin/env bash
set -e

SOURCE=/home/jdomagala/Work/SmollNet
BUILD=$SOURCE/build

CUDA=/usr/local/cuda-12.8
CLANG=/usr/local/bin

mkdir -p "$BUILD"

BUILD_TYPE=Debug
/home/jdomagala/Work/bin/conan install . -of ./build --build=missing --settings=build_type=$BUILD_TYPE -s compiler.cppstd=gnu20
if [[ "$BUILD_TYPE" == "Debug" ]]; then
  CONAN_PRESET="conan-debug"
else
  CONAN_PRESET="conan-release"
fi

cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_CXX_COMPILER="$CLANG/clang++" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="$BUILD/smollnet" \
  --preset ${CONAN_PRESET} \
  --fresh

cmake --build "$BUILD" --target install > "$BUILD/output.txt"

SOURCE=/home/jdomagala/Work/SmollNet/example
cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_CXX_COMPILER="$CLANG/clang++" \
  -DSmollNet_ROOT=${BUILD}/smollnet \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  --fresh

cmake --build "$BUILD" >> "$BUILD/output.txt"
