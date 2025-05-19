#include "helpers.hpp"
#include "kernels.cuh"

#include <cstdio>
#include <ctime>

#include <cuda.h>
#include <curand_kernel.h>

namespace smollnet {

__global__ void random_init(float *out, size_t total, size_t seed) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= total)
    return;

  curandState state;
  curand_init(seed, idx, 0, &state);

  out[idx] = curand_uniform(&state);
}

void launch_random_init(void *out, size_t total) {
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  unsigned long long seed = time(nullptr);

  random_init<<<grid, block>>>(static_cast<float *>(out), total, seed);

  CHECK_CUDA(cudaGetLastError());
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
  CHECK_CUDA(cudaGetLastError());
}

// Sum over dim-0 (collapse first index)
// Each thread computes a single output element

//   in - input data (flattened but with old dims)
//  out - output data (with new dims)
//   d0 - num elements of dim0
// rest - remaining elements
__global__ void k_sum_dim0(const float *__restrict__ in,
                           float *__restrict__ out, int64_t d0, int64_t rest) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rest)
    return;

  float acc = 0.f;

  const float *p = in + idx;

  // single thread goes over len(d0)
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
  CHECK_CUDA(cudaGetLastError());
}

__global__ void k_sum_dim1(const float *__restrict__ in,
                           float *__restrict__ out, int64_t d0, int64_t d1,
                           int64_t d2) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t n = d0 * d2; // total number of output elements
  if (idx >= n)
    return;

  // map flat idx to (i, k)
  int64_t i = idx / d2;
  int64_t k = idx % d2;

  // in[i][0][k]
  const float *p = in + i * d1 * d2 + k;
  float acc = 0.f;

  for (int64_t j = 0; j < d1; ++j, p += d2) // move along j
    acc += *p;

  out[idx] = acc;
}

void launch_sum_dim1(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {
  int64_t n = d0 * d2;
  int block = 256;
  int grid = (n + block - 1) / block;
  k_sum_dim1<<<grid, block>>>(static_cast<const float *>(in),
                              static_cast<float *>(out), d0, d1, d2);

  CHECK_CUDA(cudaGetLastError());
}

__global__ void k_sum_dim2(const float *in, float *out, int64_t outer,
                           int64_t d2) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= outer)
    return;

  // in[i][j][0]
  const float *p = in + idx * d2;
  float acc = 0.f;

  for (int64_t k = 0; k < d2; ++k)
    acc += p[k];

  out[idx] = acc;
}

void launch_sum_dim2(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {
  int64_t outer = d0 * d1;
  int block = 256;
  int grid = (outer + block - 1) / block;
  k_sum_dim2<<<grid, block>>>(static_cast<const float *>(in),
                              static_cast<float *>(out), outer, d2);

  CHECK_CUDA(cudaGetLastError());
}

__global__ void matmul_kernel(float *out, float *left, float *right, int64_t l0,
                              int64_t l1, int64_t l2, int64_t r0, int64_t r1,
                              int64_t r2, size_t total) {

  auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= total)
    return;

  int i = idx / r1;
  int j = idx % r1;

  float acc = 0.0f;
  for (int id = 0; id < l1; ++id) {
    acc += left[l1 * i + id] * right[r1 * j + id];
  }

  out[idx] = acc;
}

void launch_matmul(void *out, void *left, void *right, int64_t ldims[3],
                   int64_t rdims[3], size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  matmul_kernel<<<grid, block>>>(
      static_cast<float *>(out), static_cast<float *>(left),
      static_cast<float *>(right), ldims[0], ldims[1], ldims[2], rdims[0],
      rdims[1], rdims[2], total);

  CHECK_CUDA(cudaGetLastError());
}

} // namespace smollnet
