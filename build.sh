#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  SOURCE="$(pwd)"
else
  SOURCE="$1"
fi

BUILD=$SOURCE/build

CUDA=/usr/local/cuda-12.8
CLANG=/usr/local/bin

mkdir -p "$BUILD"

BUILD_TYPE=Release

conan profile detect
/home/jdomagala/Work/bin/conan install . -of ./build --build=missing --settings=build_type=$BUILD_TYPE -s compiler.cppstd=gnu20
if [[ "$BUILD_TYPE" == "Debug" ]]; then
  CONAN_PRESET="conan-debug"
elif [[ "$BUILD_TYPE" == "Release" ]]; then
  CONAN_PRESET="conan-release"
else
  CONAN_PRESET="conan-relwithdebinfo"
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
  --fresh | tee "$BUILD/output.txt"

cmake --build "$BUILD" --target install | tee -a "$BUILD/output.txt"

SOURCE=$SOURCE/example
cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_CXX_COMPILER="$CLANG/clang++" \
  -DSmollNet_ROOT=${BUILD}/smollnet \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
  --fresh | tee -a "$BUILD/output.txt"

cmake --build "$BUILD" | tee -a "$BUILD/output.txt"
