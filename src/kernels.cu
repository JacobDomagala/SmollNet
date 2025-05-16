#include "kernels.cuh"

#include <cuda.h>

namespace smollnet {

template <typename T> __global__ void fill_kernel(T *data, size_t n, T value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    data[idx] = value;
}

void launch_fill(float *ptr, size_t numElems, float val) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  fill_kernel<<<grid, block>>>(ptr, numElems, val);
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
}

template <typename T>
__global__ void sub_kernel(T *out, T *left, T *right, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = left[idx] - right[idx];
}

void launch_sub(float *out, float *left, float *right, size_t numElems) {
  dim3 block(256);
  dim3 grid((numElems + block.x - 1) / block.x);
  sub_kernel<<<grid, block>>>(out, left, right, numElems);
}

// sum over dim-0 (collapse first index)
__global__ void k_sum_dim0(const float *in, float *out, int64_t d0,
                           int64_t rest) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rest)
    return;

  float acc = 0.f;
  const float *p = in + idx;
  for (int64_t i = 0; i < d0; ++i, p += rest)
    acc += *p;
  out[idx] = acc;
}

void launch_sum_dim0(void *out, void *in, int64_t d0, int64_t rest) {
  int64_t n = rest;
  int block = 256;
  int grid = (n + block - 1) / block;
  k_sum_dim0<<<grid, block>>>(static_cast<const float *>(in),
                              static_cast<float *>(out), d0, rest);
}

__global__ void k_sum_dim1(const float *in, float *out, int64_t d0,
                           int64_t rest) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rest)
    return;

  float acc = 0.f;
  const float *p = in + idx;
  for (int64_t i = 0; i < d0; ++i, p += rest)
    acc += *p;
  out[idx] = acc;
}

void launch_sum_dim1(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {
  int64_t n = d0 * d2;
  int block = 256;
  int grid = (n + block - 1) / block;
  k_sum_dim1<<<grid, block>>>(static_cast<const float *>(in),
                              static_cast<float *>(out), d0, n);
}

__global__ void k_sum_dim2(const float *in, float *out, int64_t d0,
                           int64_t rest) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rest)
    return;

  float acc = 0.f;
  const float *p = in + idx;
  for (int64_t i = 0; i < d0; ++i, p += rest)
    acc += *p;
  out[idx] = acc;
}

void launch_sum_dim2(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {
  int64_t n = d0 * d2;
  int block = 256;
  int grid = (n + block - 1) / block;
  k_sum_dim2<<<grid, block>>>(static_cast<const float *>(in),
                              static_cast<float *>(out), d0, n);
}

} // namespace smollnet
