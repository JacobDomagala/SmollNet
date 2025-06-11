#include "helpers.hpp"
#include "kernels.cuh"

#include <cstdio>
#include <ctime>

#include <cuda.h>
#include <curand_kernel.h>

namespace smollnet {
__device__ __forceinline__ void compute_dimensions(int (&dims)[3], size_t idx,
                                                   const StrideInfo &s) {

  if (s.rank == 3) {
    int64_t rest = s.size[1] * s.size[2];
    dims[0] = idx / rest;
    int64_t rem = idx % rest;
    dims[1] = rem / s.size[2];
    dims[2] = rem % s.size[2];
  } else if (s.rank == 2) {
    dims[0] = idx / s.size[1];
    dims[1] = idx % s.size[1];
    dims[2] = 0;
  } else { // rank == 1
    dims[0] = idx;
    dims[1] = 0;
    dims[2] = 0;
  }
}

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

  // Decode linear index -> (i,j,k)
  int dims[3] = {0, 0, 0};
  compute_dimensions(dims, idx, s);

  int64_t offA =
      dims[0] * s.astr[0] + dims[1] * s.astr[1] + dims[2] * s.astr[2];
  int64_t offB =
      dims[0] * s.bstr[0] + dims[1] * s.bstr[1] + dims[2] * s.bstr[2];

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

  // Decode linear index -> (i,j,k)
  int dims[3] = {0, 0, 0};
  compute_dimensions(dims, idx, s);

  int64_t offA =
      dims[0] * s.astr[0] + dims[1] * s.astr[1] + dims[2] * s.astr[2];
  int64_t offB =
      dims[0] * s.bstr[0] + dims[1] * s.bstr[1] + dims[2] * s.bstr[2];

  out[idx] = a[offA] - b[offB];
}
void launch_sub_strided(void *out, void *a, void *b, const StrideInfo &s,
                        size_t total) {
  dim3 block = 256;
  dim3 grid = (total + block.x - 1) / block.x;
  sub_strided_kernel<<<grid, block>>>(static_cast<float *>(out),
                                      static_cast<float *>(a),
                                      static_cast<float *>(b), s, total);
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
    acc += left[l1 * i + id] * right[r1 * id + j];
  }

  out[idx] = acc;
}

void launch_matmul(void *out, void *left, void *right, const int64_t ldims[3],
                   const int64_t rdims[3], size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  matmul_kernel<<<grid, block>>>(
      static_cast<float *>(out), static_cast<float *>(left),
      static_cast<float *>(right), ldims[0], ldims[1], ldims[2], rdims[0],
      rdims[1], rdims[2], total);

  CHECK_CUDA(cudaGetLastError());
}

__global__ void relu_kernel(float *out, float *in, size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total)
    out[idx] = max(in[idx], 0.0f);
}

void launch_relu(void *out, void *in, size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  relu_kernel<<<grid, block>>>(static_cast<float *>(out),
                               static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void relu_grad_kernel(float *out, float *grad_out, float *in,
                                 size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total)
    out[idx] = (in[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

void launch_relu_grad(void *out, void *grad_out, void *in, size_t total) {
  int block = 256;
  int grid = (total + block - 1) / block;

  relu_grad_kernel<<<grid, block>>>(static_cast<float *>(out),
                                    static_cast<float *>(grad_out),
                                    static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void gelu_kernel(float *out, float *in, size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    out[idx] =
        0.5f * in[idx] *
        (1.0f + tanhf(sqrt_2_over_pi *
                      (in[idx] + 0.044715f * in[idx] * in[idx] * in[idx])));
  }
}

void launch_gelu(void *out, void *in, size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  gelu_kernel<<<grid, block>>>(static_cast<float *>(out),
                               static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void gelu_grad_kernel(float *out, float *grad_out, float *in,
                                 size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total) {
    // const float a = std::sqrt(2.0f / M_PI);
    constexpr float a = 0.7978845608f;
    constexpr float b = 0.044715f;
    float x3 = in[idx] * in[idx] * in[idx];
    float h = in[idx] + b * x3;
    float tanh_ax = tanhf(a * h);
    float sech2 = 1.0f - tanh_ax * tanh_ax;
    float h_prime = 1.0f + 3.0f * b * in[idx] * in[idx];

    float g = 0.5f * (1.0f + tanh_ax) + 0.5f * in[idx] * sech2 * a * h_prime;
    out[idx] = grad_out[idx] * g;
  }
}

void launch_gelu_grad(void *out, void *grad_out, void *in, size_t total) {
  int block = 256;
  int grid = (total + block - 1) / block;

  gelu_grad_kernel<<<grid, block>>>(static_cast<float *>(out),
                                    static_cast<float *>(grad_out),
                                    static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void tanh_kernel(float *out, float *in, size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total)
    out[idx] = tanhf(in[idx]);
}

void launch_tanh(void *out, void *in, size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  tanh_kernel<<<grid, block>>>(static_cast<float *>(out),
                               static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void tanh_grad_kernel(float *out, float *grad_out, float *in,
                                 size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total)
    out[idx] = grad_out[idx] * (1.0f - in[idx] * in[idx]);
}

void launch_tanh_grad(void *out, void *grad_out, void *in, size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  tanh_grad_kernel<<<grid, block>>>(static_cast<float *>(out),
                                    static_cast<float *>(grad_out),
                                    static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void sigmoid_kernel(float *output, float *input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] =
        1.0f / (1.0f + expf(-input[idx])); // Apply sigmoid to each element
  }
}

void launch_sigmoid(void *out, void *in, size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  sigmoid_kernel<<<grid, block>>>(static_cast<float *>(out),
                                  static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void sigmoid_grad_kernel(float *output, float *grad_out,
                                    float *input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = grad_out[idx] * input[idx] * (1.0f - input[idx]);
  }
}

void launch_sigmoid_grad(void *out, void *grad_out, void *in, size_t total) {

  int block = 256;
  int grid = (total + block - 1) / block;

  sigmoid_grad_kernel<<<grid, block>>>(static_cast<float *>(out),
                                       static_cast<float *>(grad_out),
                                       static_cast<float *>(in), total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void mse_square_kernel(float *out, float *pred, float *target,
                                  size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= total)
    return;

  out[idx] = powf((pred[idx] - target[idx]), 2.0f);
}

__global__ void mse_sum_kernel(float *out, float *in, size_t N) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx > 0)
    return;

  float acc = 0.0f;
  for (int i = 0; i < N; ++i) {
    acc += in[i];
  }

  out[idx] = acc / N;
}

void launch_mse(void *out, void *pred, void *target, size_t total) {
  int block = 256;
  int grid = (total + block - 1) / block;

  mse_square_kernel<<<grid, block>>>(static_cast<float *>(out),
                                     static_cast<float *>(pred),
                                     static_cast<float *>(target), total);

  CHECK_CUDA(cudaGetLastError());

  mse_sum_kernel<<<grid, block>>>(static_cast<float *>(out),
                                  static_cast<float *>(out), total);

  CHECK_CUDA(cudaGetLastError());
}
__global__ void sgd_kernel(float *w, const float *grad, float lr,
                           size_t total) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    w[idx] -= lr * grad[idx];
  }
}

void launch_sgd_update(void *p, void *g, float lr, size_t total) {
  dim3 block = 256;
  dim3 grid = (total + block.x - 1) / block.x;
  sgd_kernel<<<grid, block>>>(static_cast<float *>(p),
                              static_cast<const float *>(g), lr, total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void mse_grad_kernel(float *g, const float *p, const float *t,
                                float coeff, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    g[idx] = coeff * (p[idx] - t[idx]);
}

void launch_mse_grad(void *grad, void *pred, void *target, float coeff,
                     size_t total) {
  int block = 256;
  int grid = (total + block - 1) / block;
  mse_grad_kernel<<<grid, block>>>(static_cast<float *>(grad),
                                   static_cast<float *>(pred),
                                   static_cast<float *>(target), coeff, total);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void
variance_step1_kernel(float* out, float* in, const float mean, const size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx >= total) return;

  out[idx] = powf(mean - in[idx], 2);
}

// void launch_variance(void* out, void* in, float mean, size_t total) {
//   dim3 block = 256;
//   dim3 grid = (block.x + total - 1) / block.x;

//   variance_step1_kernel<<<grid, block>>>(static_cast<float*>(out), static_cast<float*>(in), mean, total);

//   k_sum_dim1<<<1, total>>>(const float *__restrict in, float *__restrict out, int64_t d0, int64_t d1, int64_t d2)
// }

} // namespace smollnet
