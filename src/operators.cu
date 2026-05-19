#include "dtype_utils.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <cuda.h>

namespace smollnet {

namespace {

struct AddOp {
  __device__ float operator()(float lhs, float rhs) const { return lhs + rhs; }
};

struct SubOp {
  __device__ float operator()(float lhs, float rhs) const { return lhs - rhs; }
};

struct MulOp {
  __device__ float operator()(float lhs, float rhs) const { return lhs * rhs; }
};

struct DivOp {
  __device__ float operator()(float lhs, float rhs) const { return lhs / rhs; }
};

__device__ __forceinline__ void compute_strided_offsets(size_t idx,
                                                        const StrideInfo &s,
                                                        int64_t &offA,
                                                        int64_t &offB) {
  offA = 0;
  offB = 0;

  for (int64_t dim = s.rank - 1; dim >= 0; --dim) {
    const int64_t coord = idx % s.output_size[dim];
    idx /= s.output_size[dim];

    offA += coord * s.a_stride[dim];
    offB += coord * s.b_stride[dim];
  }
}

template <typename T> __global__ void negative_kernel(T *ptr, size_t total) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < total) {
    ptr[idx] = scalar_from_float<T>(-scalar_to_float(ptr[idx]));
  }
}

template <typename T> __global__ void fill_kernel(T *data, size_t n,
                                                  float value) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = scalar_from_float<T>(value);
  }
}

template <typename T, typename Op>
__global__ void binary_kernel(T *out, const T *left, const T *right, size_t n,
                              Op op) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float lhs = scalar_to_float(left[idx]);
    const float rhs = scalar_to_float(right[idx]);
    out[idx] = scalar_from_float<T>(op(lhs, rhs));
  }
}

template <typename T, typename Op>
__global__ void binary_strided_kernel(T *__restrict__ out,
                                      const T *__restrict__ a,
                                      const T *__restrict__ b, StrideInfo s,
                                      size_t total, Op op) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t offA = 0;
  int64_t offB = 0;
  compute_strided_offsets(idx, s, offA, offB);

  const float lhs = scalar_to_float(a[offA]);
  const float rhs = scalar_to_float(b[offB]);
  out[idx] = scalar_from_float<T>(op(lhs, rhs));
}

template <typename Op>
void launch_binary(void *out, const void *left, const void *right,
                   DataType dtype, size_t numElems, Op op) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);

  dispatch_float_dtype(dtype, [&]<typename T>() {
    binary_kernel<T, Op><<<grid, block>>>(static_cast<T *>(out),
                                          static_cast<const T *>(left),
                                          static_cast<const T *>(right),
                                          numElems, op);
  });

  CHECK_CUDA(cudaGetLastError());
}

template <typename Op>
void launch_binary_strided(void *out, const void *left, const void *right,
                           DataType dtype, const StrideInfo &s, size_t total,
                           Op op) {
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);

  dispatch_float_dtype(dtype, [&]<typename T>() {
    binary_strided_kernel<T, Op><<<grid, block>>>(
        static_cast<T *>(out), static_cast<const T *>(left),
        static_cast<const T *>(right), s, total, op);
  });

  CHECK_CUDA(cudaGetLastError());
}

} // namespace

void launch_negative(void *ptr, DataType dtype, size_t total) {
  dim3 block = 256;
  dim3 grid = (block.x + total - 1) / block.x;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    negative_kernel<T><<<grid, block>>>(static_cast<T *>(ptr), total);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_fill(void *ptr, DataType dtype, size_t numElems, float val) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);

  dispatch_float_dtype(dtype, [&]<typename T>() {
    fill_kernel<T><<<grid, block>>>(static_cast<T *>(ptr), numElems, val);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_add(void *out, const void *left, const void *right, DataType dtype,
                size_t numElems) {
  launch_binary(out, left, right, dtype, numElems, AddOp{});
}

void launch_add_strided(void *dst, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total) {
  launch_binary_strided(dst, a, b, dtype, s, total, AddOp{});
}

void launch_mul(void *out, const void *left, const void *right, DataType dtype,
                size_t numElems) {
  launch_binary(out, left, right, dtype, numElems, MulOp{});
}

void launch_mul_strided(void *dst, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total) {
  launch_binary_strided(dst, a, b, dtype, s, total, MulOp{});
}

void launch_sub(void *out, const void *a, const void *b, DataType dtype,
                size_t numElems) {
  launch_binary(out, a, b, dtype, numElems, SubOp{});
}

void launch_sub_strided(void *out, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total) {
  launch_binary_strided(out, a, b, dtype, s, total, SubOp{});
}

void launch_div(void *out, const void *a, const void *b, DataType dtype,
                size_t numElems) {
  launch_binary(out, a, b, dtype, numElems, DivOp{});
}

void launch_div_strided(void *dst, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total) {
  launch_binary_strided(dst, a, b, dtype, s, total, DivOp{});
}

} // namespace smollnet
