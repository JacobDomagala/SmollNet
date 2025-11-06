#include "helpers.hpp"
#include "kernels.cuh"

#include <cuda.h>

namespace smollnet {

__device__ __forceinline__ void compute_dimensions(int (&dims)[3], size_t idx,
                                                   const StrideInfo &s) {

  if (s.rank == 3) {
    int64_t rest = s.output_size[1] * s.output_size[2];
    dims[0] = idx / rest;
    int64_t rem = idx % rest;
    dims[1] = rem / s.output_size[2];
    dims[2] = rem % s.output_size[2];
  } else if (s.rank == 2) {
    dims[0] = idx / s.output_size[1];
    dims[1] = idx % s.output_size[1];
    dims[2] = 0;
  } else { // rank == 1
    dims[0] = idx;
    dims[1] = 0;
    dims[2] = 0;
  }
}

__global__ void negative_kernel(float *ptr, size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < total)
    ptr[idx] *= -1.0f;
}

void launch_negative(void *ptr, size_t total) {
  dim3 block = 256;
  dim3 grid = (block.x + total - 1) / block.x;

  negative_kernel<<<grid, block>>>(static_cast<float *>(ptr), total);
}

template <typename T> __global__ void fill_kernel(T *data, size_t n, T value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    data[idx] = value;
}

void launch_fill(float *ptr, size_t numElems, float val) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  fill_kernel<<<grid, block>>>(ptr, numElems, val);
  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__global__ void add_kernel(T *out, T *left, T *right, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = left[idx] + right[idx];
}

void launch_add(float *out, float *left, float *right, size_t numElems) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  add_kernel<<<grid, block>>>(out, left, right, numElems);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void add_strided_kernel(float *__restrict__ out,
                                   const float *__restrict__ a,
                                   const float *__restrict__ b, StrideInfo s,
                                   size_t total) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  int dims[3] = {0, 0, 0};
  compute_dimensions(dims, idx, s);

  int64_t offA = dims[0] * s.a_stride[0] + dims[1] * s.a_stride[1] +
                 dims[2] * s.a_stride[2];
  int64_t offB = dims[0] * s.b_stride[0] + dims[1] * s.b_stride[1] +
                 dims[2] * s.b_stride[2];

  out[idx] = a[offA] + b[offB];
}

void launch_add_strided(void *dst, void *a, void *b, const StrideInfo &s,
                        size_t total) {
  dim3 blk(256);
  dim3 grd((total + blk.x - 1) / blk.x);

  add_strided_kernel<<<grd, blk>>>(static_cast<float *>(dst),
                                   static_cast<const float *>(a),
                                   static_cast<const float *>(b), s, total);
  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__global__ void mul_kernel(T *out, T *left, T scalar, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = left[idx] * scalar;
}

template <typename T>
__global__ void mul_kernel(T *out, T *left, T *right, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = left[idx] * right[idx];
}

void launch_mul(float *out, float *left, float *right, size_t numElems) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  mul_kernel<<<grid, block>>>(out, left, right, numElems);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void mul_strided_kernel(float *__restrict__ out,
                                   const float *__restrict__ a,
                                   const float *__restrict__ b, StrideInfo s,
                                   size_t total) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  int dims[3] = {0, 0, 0};
  compute_dimensions(dims, idx, s);

  int64_t offA = dims[0] * s.a_stride[0] + dims[1] * s.a_stride[1] +
                 dims[2] * s.a_stride[2];
  int64_t offB = dims[0] * s.b_stride[0] + dims[1] * s.b_stride[1] +
                 dims[2] * s.b_stride[2];

  out[idx] = a[offA] * b[offB];
}

void launch_mul_strided(void *dst, void *a, void *b, const StrideInfo &s,
                        size_t total) {
  dim3 blk(256);
  dim3 grd((total + blk.x - 1) / blk.x);

  mul_strided_kernel<<<grd, blk>>>(static_cast<float *>(dst),
                                   static_cast<const float *>(a),
                                   static_cast<const float *>(b), s, total);
  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__global__ void sub_kernel(T *out, T *left, T *right, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = left[idx] - right[idx];
}

void launch_sub(float *out, float *a, float *b, size_t numElems) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  sub_kernel<<<grid, block>>>(out, a, b, numElems);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void sub_strided_kernel(float *out, float *a, float *b, StrideInfo s,
                                   size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= total)
    return;

  int dims[3] = {0, 0, 0};
  compute_dimensions(dims, idx, s);

  int64_t offA = dims[0] * s.a_stride[0] + dims[1] * s.a_stride[1] +
                 dims[2] * s.a_stride[2];
  int64_t offB = dims[0] * s.b_stride[0] + dims[1] * s.b_stride[1] +
                 dims[2] * s.b_stride[2];

  out[idx] = a[offA] - b[offB];
}

void launch_sub_strided(void *out, void *a, void *b, const StrideInfo &s,
                        size_t total) {
  dim3 block = 256;
  dim3 grid = (total + block.x - 1) / block.x;
  sub_strided_kernel<<<grid, block>>>(static_cast<float *>(out),
                                      static_cast<float *>(a),
                                      static_cast<float *>(b), s, total);
  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__global__ void div_kernel(T *out, T *left, T *right, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = left[idx] / right[idx];
}

void launch_div(float *out, float *a, float *b, size_t numElems) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  div_kernel<<<grid, block>>>(out, a, b, numElems);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void div_strided_kernel(float *__restrict__ out,
                                   const float *__restrict__ a,
                                   const float *__restrict__ b, StrideInfo s,
                                   size_t total) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  int dims[3] = {0, 0, 0};
  compute_dimensions(dims, idx, s);

  int64_t offA = dims[0] * s.a_stride[0] + dims[1] * s.a_stride[1] +
                 dims[2] * s.a_stride[2];
  int64_t offB = dims[0] * s.b_stride[0] + dims[1] * s.b_stride[1] +
                 dims[2] * s.b_stride[2];

  out[idx] = a[offA] / b[offB];
}

void launch_div_strided(void *dst, void *a, void *b, const StrideInfo &s,
                        size_t total) {
  dim3 blk(256);
  dim3 grd((total + blk.x - 1) / blk.x);

  div_strided_kernel<<<grd, blk>>>(static_cast<float *>(dst),
                                   static_cast<const float *>(a),
                                   static_cast<const float *>(b), s, total);
  CHECK_CUDA(cudaGetLastError());
}

} // namespace smollnet
