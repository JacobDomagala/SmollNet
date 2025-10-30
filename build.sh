#!/usr/bin/env bash
set -e

if [ -t 1 ]; then
  BLUE=$'\033[1;34m'
  CYAN=$'\033[1;36m'
  RESET=$'\033[0m'
else
  BLUE=""
  CYAN=""
  RESET=""
fi

info() {
  printf "%b==>%b %s\n" "$BLUE" "$RESET" "$1"
}

info_value() {
  printf "  %b%s%b %s\n" "$CYAN" "$1:" "$RESET" "$2"
}

SOURCE="${1:-$(pwd)}"
BUILD="$SOURCE/build"
mkdir -p "$BUILD"

BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda-12.8}}"
CUDA_COMPILER="${CUDA_COMPILER:-${CMAKE_CUDA_COMPILER:-$CUDA_ROOT/bin/nvcc}}"
CXX_COMPILER="${CXX_COMPILER:-${CMAKE_CXX_COMPILER:-${CXX:-clang++}}}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-89}"

info "Build configuration"
info_value "CUDA root" "$CUDA_ROOT"
info_value "CUDA compiler" "$CUDA_COMPILER"
info_value "C++ compiler" "$CXX_COMPILER"
info_value "CUDA architectures" "$CUDA_ARCHITECTURES"

if ! conan profile list | grep -q "default"; then
    info "Conan 'default' profile not found. Detecting and creating it..."
    conan profile detect
else
    info "Conan 'default' profile already exists. Skipping detection."

fi
conan install "$SOURCE" -of "$BUILD" --build=missing --settings=build_type="$BUILD_TYPE" -s compiler.cppstd=gnu23
if [[ "$BUILD_TYPE" == "Debug" ]]; then
  CONAN_PRESET="conan-debug"
elif [[ "$BUILD_TYPE" == "Release" ]]; then
  CONAN_PRESET="conan-release"
else
  CONAN_PRESET="conan-relwithdebinfo"
fi

cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="$BUILD/smollnet" \
  --preset "$CONAN_PRESET" \
  --fresh | tee "$BUILD/output.txt"

cmake --build "$BUILD" --target install | tee -a "$BUILD/output.txt"

SOURCE=$SOURCE/example
cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DSmollNet_ROOT="${BUILD}/smollnet" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
  --fresh | tee -a "$BUILD/output.txt"

cmake --build "$BUILD" | tee -a "$BUILD/output.txt"
