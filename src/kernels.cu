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

  // Decode linear index -> (i,j,k)
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

  // Decode linear index -> (i,j,k)
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
}

template <int AXIS>
__global__ void sum_kernel(const float *__restrict__ in,
                           float *__restrict__ out, int64_t d0, int64_t d1,
                           int64_t d2) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Remaining elements -> each elem will compute
  const int64_t num_elems = AXIS == 0 ? d1 * d2 : AXIS == 1 ? d0 * d2 : d0 * d1;

  // const int64_t num_elems = d0 * d1 * d2;

  extern __shared__ float sMem[];

  if (idx >= num_elems)
    return;

  int64_t stride = 0;
  int64_t i0 = 0;
  int64_t i1 = 0;
  int64_t i2 = 0;
  int64_t base = 0;
  int64_t size = 0;

  if constexpr (AXIS == 0) {
    stride = d1 * d2;
    i0 = idx / stride;
    i1 = idx / d2;
    i2 = idx % d2;
    size = d0;
    base = stride;
  } else if constexpr (AXIS == 1) {

  } else {
  }

  const float *p = in + idx;
  float acc = 0.f;

#pragma unroll
  for (int i = 0; i < size; ++i) {
    acc += *p;
    p += stride;
  }

  out[idx] = acc;
}

template <int AXIS>
__global__ void sum_kernel_atomic(const float *__restrict__ in,
                                  float *__restrict__ out, int64_t d0,
                                  int64_t d1, int64_t d2) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Remaining elements -> each elem will compute
  // const int64_t num_elems = AXIS == 0 ? d1 * d2 : AXIS == 1 ? d0 * d2 : d0 *
  // d1;

  const int64_t num_elems = d0 * d1 * d2;

  extern __shared__ float sMem[];

  if (idx >= num_elems)
    return;

  int64_t stride = 0;
  int64_t i0 = 0;
  int64_t i1 = 0;
  int64_t i2 = 0;
  int64_t base = 0;
  int64_t size = 0;

  if constexpr (AXIS == 0) {
    stride = d1 * d2;
    i0 = idx / stride;
    i1 = (idx / d2) % d1;
    i2 = idx % d2;
    size = d0;
    base = stride;
  } else if constexpr (AXIS == 1) {

  } else {
  }

  const float *p = in + idx;

  atomicAdd(out + (i1 * d2 + i2), *p);
}

template <unsigned int BLOCK_SIZE>
__global__ void reduction_kernel(const float *__restrict__ in,
                                 float *__restrict__ out, const int64_t d0,
                                 const int64_t d1, const int64_t d2) {
  // Each block handles one output element at index (i1, i2)
  // The grid should be launched as dim3(d2, d1)
  const int64_t i1 = blockIdx.y;
  const int64_t i2 = blockIdx.x;

  // The output memory location for this block
  const int64_t out_idx = i1 * d2 + i2;

  // This thread's partial sum
  float my_sum = 0.0f;

  // Stride between consecutive elements along the reduction axis (d0)
  const int64_t reduction_stride = d1 * d2;

  // Each thread in the block cooperates to reduce the slice.
  // We start at an offset corresponding to this thread's lane in the block.
  // The loop strides by the block size, so the block as a whole sweeps
  // through the entire slice along the d0 axis.
  for (int64_t i = threadIdx.x; i < d0; i += BLOCK_SIZE) {
    // *** CORRECT COALESCED READ ***
    // Consecutive threads (threadIdx.x, threadIdx.x+1, ...) access
    // consecutive memory locations because 'out_idx' is constant for the block
    // and 'i' increases by 1 for each thread.
    my_sum += in[out_idx + i * reduction_stride];
  }

  // --- In-block reduction using shared memory ---
  extern __shared__ float sMem[];
  sMem[threadIdx.x] = my_sum;
  __syncthreads();

  // Standard parallel reduction within the shared memory tile
  for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sMem[threadIdx.x] += sMem[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Thread 0 of the block writes the final, reduced sum for this slice.
  // No atomics are needed because each block writes to a unique location.
  if (threadIdx.x == 0) {
    out[out_idx] = sMem[0];
  }
}

// __global__ void
// tiled_sum_2d(const float* in, float* out, size_t d0, size_t d1, size_t d2) {

//   const int64_t idx = threadIdx.x + blockDim.x * blockIdx.x
//   // 16x16 tile and also block
//   extern __shared__ float sMem[];

//   size_t stride = 0;

//   // transposed manner so warp can local reduce
//   sMem[threadIdx.y + threadIdx.x * blockDim.y] = in[threadIdx.x + threadIdx.y
//   * stride];

//   __syncthreads();

//   float v = 0.0f;
//   #pragma unroll
//   for (int off = 16; off > 0; off >>= 1)
//     v += __shfl_down_sync(0xffffffff, v, off);

//   if ((threadIdx.x & 31) == 0)
//     atomicAdd(out + offset, v);
// }

template <typename T, unsigned int BLOCK_SIZE = 256, unsigned int VEC = 4>
__global__ void reduce_axis0_kernel(const T *__restrict__ in,
                                    T *__restrict__ out, size_t N, size_t M) {
  const size_t colStart = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * VEC;
  if (colStart >= M)
    return;

  T accum[VEC];
#pragma unroll
  for (int v = 0; v < VEC; ++v)
    accum[v] = T(0);

  for (size_t row = 0; row < N; ++row) {
    const size_t base = row * M + colStart;
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      const size_t col = colStart + v;
      if (col < M)
        accum[v] += in[base + v];
    }
  }

#pragma unroll
  for (int v = 0; v < VEC; ++v) {
    const size_t col = colStart + v;
    if (col < M)
      out[col] = accum[v];
  }
}

__global__ void warp_level_sum(const float *__restrict__ in,
                               float *__restrict__ out, size_t dim_size,
                               size_t offset,
                               size_t n) // n = d0*d1*d2
{
  // We always launch this kernel as 1D block, the only difference can be a grid
  // dim
  int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t i = blockIdx.y;
  int64_t d = blockIdx.z;

  int64_t idx = d * offset + i * dim_size + j;
  int64_t out_idx = d * offset + i;

  float v = (idx < n and j < dim_size) ? in[idx] : 0.0f;

  extern __shared__ float sMem[];

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);

  if ((threadIdx.x & 31) == 0 and j < dim_size) {
    sMem[threadIdx.x / 32] = v;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    float acc = 0.0f;
    for (int i = 0; i < blockDim.x / 32; ++i) {
      acc += sMem[i];
    }
    atomicAdd(out + out_idx, acc);
  }
}

void launch_sum_dim0(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {

  // Contigious memory access -> warp level reduce!
  if (d1 * d2 == 1) {
    dim3 block = 256;
    const auto total = d0 * d1 * d2;
    dim3 grid = (total + 256 - 1) / 256;
    warp_level_sum<<<grid, block, (256 / 32) * sizeof(float)>>>(
        static_cast<const float *>(in), static_cast<float *>(out), d0, 0,
        total);
  } else {
    // For strided pattern use tile + smem
  }
}

void launch_sum_dim1(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {
  // Contigious memory access -> warp level reduce!
  if (d2 == 1) {
    dim3 block(256, 1, 1);
    const auto total = d0 * d1 * d2;
    dim3 grid((d1 + block.x - 1) / block.x, d0, 1);
    warp_level_sum<<<grid, block, (256 / 32) * sizeof(float)>>>(
        static_cast<const float *>(in), static_cast<float *>(out), d1, 0,
        total);
  } else {
    // For strided pattern use tile + smem
    int block = 256;
    int grid = (d1 + block - 1) / block;
    sum_kernel<1><<<grid, block, d1 * sizeof(float)>>>(
        static_cast<const float *>(in), static_cast<float *>(out), d0, d1, d2);
  }

  CHECK_CUDA(cudaGetLastError());
}

void launch_sum_dim2(void *out, void *in, int64_t d0, int64_t d1, int64_t d2) {
  // Since we only support Rank3 Tensors, this is always contigious memory
  dim3 block(256, 1, 1);
  const auto total = d0 * d1 * d2;
  dim3 grid((block.x + d2 - 1) / block.x, d1, d0);

  fmt::print(
      "Launching warp_level_sum<<<({},{},{}), ({},{},{})>>>(in,out,{},{},{})\n",
      grid.x, grid.y, grid.z, block.x, block.y, block.z, d2, d0 * d1, total);
  warp_level_sum<<<grid, block, (256 / 32) * sizeof(float)>>>(
      static_cast<const float *>(in), static_cast<float *>(out), d2, d1, total);

  CHECK_CUDA(cudaGetLastError());
}

__global__ void matmul_kernel(float *__restrict__ C,
                              const float *__restrict__ A,
                              const float *__restrict__ B,
                              const StrideInfo strides, const SizeInfo sizes,
                              const int tile_width) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x; // N‑index
  const int row = blockIdx.y * blockDim.y + threadIdx.y; // M‑index

  const int M = strides.output_size[0];
  const int N = strides.output_size[1];
  const int K = sizes.a_size[1]; // = sizes.b_size[0]

  const bool in_bounds = (row < M) && (col < N);

  extern __shared__ float s_mem[];
  float *As = s_mem;                           // tile from A (M×K)
  float *Bs = s_mem + tile_width * tile_width; // tile from B (K×N)

  float acc = 0.0f;
  const int num_tiles = (K + tile_width - 1) / tile_width;

  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * tile_width + threadIdx.x; // K‑index into A
    const int b_row = t * tile_width + threadIdx.y; // K‑index into B

    // Load current tiles into shared memory, zero‑padding out‑of‑range
    // elements.
    As[threadIdx.y * tile_width + threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

    Bs[threadIdx.y * tile_width + threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    // Multiply–accumulate over the valid fragment length.
    const int elems = min(tile_width, K - t * tile_width);
#pragma unroll
    for (int e = 0; e < elems; ++e)
      acc +=
          As[threadIdx.y * tile_width + e] * Bs[e * tile_width + threadIdx.x];

    __syncthreads();
  }

  if (in_bounds)
    C[row * N + col] = acc;
}

void launch_matmul(void *out, void *left, void *right,
                   const StrideInfo &strides, const SizeInfo &sizes,
                   size_t total) {
  constexpr int TILE = 16;
  dim3 block(TILE, TILE);

  const int M = strides.output_size[0]; // rows of C
  const int N = strides.output_size[1]; // cols of C

  dim3 grid((N + TILE - 1) / TILE,  // x‑dim ← N
            (M + TILE - 1) / TILE); // y‑dim ← M

  size_t smem_bytes = 2 * TILE * TILE * sizeof(float);

  matmul_kernel<<<grid, block, smem_bytes>>>(
      static_cast<float *>(out), static_cast<const float *>(left),
      static_cast<const float *>(right), strides, sizes, TILE);

  CHECK_CUDA(cudaGetLastError());
}

__global__ void relu_kernel(float *out, float *in, size_t total) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total)
    out[idx] = fmaxf(in[idx], 0.0f);
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

__global__ void mse_kernel(float *out, const float *__restrict__ pred,
                           const float *__restrict__ target, std::size_t n) {
  extern __shared__ float sdata[];
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  float local_sum = 0.0f;

  for (; idx < n; idx += stride) {
    float diff = pred[idx] - target[idx];
    local_sum += diff * diff;
  }

  sdata[threadIdx.x] = local_sum;
  __syncthreads();

  // reduction in shared memory
  for (std::size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    atomicAdd(out, sdata[0]);
}

void launch_mse(void *out, void *pred, void *target, size_t total) {
  int block = 256;
  int grid = (total + block - 1) / block;

  mse_kernel<<<grid, block, block * sizeof(float)>>>(
      static_cast<float *>(out), static_cast<float *>(pred),
      static_cast<float *>(target), total);

  mul_kernel<<<1, 1>>>(static_cast<float *>(out), static_cast<float *>(out),
                       1.0f / static_cast<float>(total), 1);
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

__global__ void variance_step1_kernel(float *out, float *in, float *mean,
                                      const size_t batch_size,
                                      const size_t num_features) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= batch_size * num_features)
    return;

  int batch_num = idx / num_features;
  out[idx] = powf(mean[batch_num] - in[idx], 2);
}

__global__ void variance_step2_kernel(float *out, float *in,
                                      const size_t batch_size,
                                      const size_t num_features) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= batch_size)
    return;

  float acc = 0.0f;

  for (int i = 0; i < num_features; ++i) {
    acc += in[idx * num_features + i];
  }

  acc /= num_features;

  out[idx] = acc;
}

void launch_variance(void *variance, void *staging_buffer, void *in, void *mean,
                     size_t batch_size, size_t num_features) {
  dim3 block = 256;
  dim3 grid = (block.x + batch_size * num_features - 1) / block.x;

  variance_step1_kernel<<<grid, block>>>(
      static_cast<float *>(staging_buffer), static_cast<float *>(in),
      static_cast<float *>(mean), batch_size, num_features);

  variance_step2_kernel<<<grid, block>>>(static_cast<float *>(variance),
                                         static_cast<float *>(staging_buffer),
                                         batch_size, num_features);
}

__global__ void mean_2d_kernel(float *out, float *in, size_t d0, size_t d1) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= d0)
    return;

  float acc = 0.0f;
  for (int i = 0; i < d1; ++i) {
    acc += in[idx * d1 + i];
  }

  acc /= d1;

  out[idx] = acc;
}

void launch_mean_2d(void *out, void *in, size_t d0, size_t d1) {
  dim3 block = 256;
  dim3 grid = (block.x + d0 * d1 - 1) / block.x;

  mean_2d_kernel<<<grid, block>>>(static_cast<float *>(out),
                                  static_cast<float *>(in), d0, d1);
}

__global__ void layer_norm_kernel(float *out, float *features, float *mean,
                                  float *variance, float *gamma, float *beta,
                                  size_t batch_size, size_t num_features) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto total = batch_size * num_features;

  if (idx >= total)
    return;

  int batch_num = idx / num_features;

  constexpr float epsilon = 1e-5f;
  float normalized =
      (features[idx] - mean[batch_num]) / sqrtf(variance[batch_num] + epsilon);

  out[idx] = gamma[batch_num] * normalized + beta[batch_num];
}

void launch_layer_norm(void *out, void *features, void *mean, void *variance,
                       void *gamma, void *beta, size_t batch_size,
                       size_t num_features) {
  dim3 block = 256;
  size_t total = batch_size * num_features;
  dim3 grid = (block.x + total - 1) / block.x;

  layer_norm_kernel<<<grid, block>>>(
      static_cast<float *>(out), static_cast<float *>(features),
      static_cast<float *>(mean), static_cast<float *>(variance),
      static_cast<float *>(gamma), static_cast<float *>(beta), batch_size,
      num_features);
}

__global__ void layer_norm_grad_kernel(float *out_grad,
                                       const float *normalized_input,
                                       const float *scaled_gradient,
                                       const float *variance,
                                       const float *summed_scale,
                                       const float *summed_scaled_input,
                                       size_t batch_size, size_t num_features) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t total = batch_size * num_features;
  if (idx >= total)
    return;

  const size_t row = idx / num_features;

  constexpr float eps = 1e-5f;
  const float inv_std = rsqrtf(variance[row] + eps); // per-sample variance
  const float m1 = summed_scale[row] / num_features; // Σδ / D
  const float m2 = summed_scaled_input[row] / num_features; // Σδ·ẋ / D

  const float hat_x = normalized_input[idx];
  const float delta = scaled_gradient[idx]; // δ = dy * γ

  const float res = inv_std * (delta - m1 - hat_x * m2); // ∂L/∂x
  out_grad[idx] = res;
}

void launch_layer_norm_grad(void *out, void *normalized_input,
                            void *scaled_gradient, void *variance,
                            void *summed_scale, void *summed_scaled_input,
                            size_t batch_size, size_t num_features) {

  dim3 block = 256;
  size_t total = batch_size * num_features;
  dim3 grid = (block.x + total - 1) / block.x;
  layer_norm_grad_kernel<<<grid, block>>>(
      static_cast<float *>(out), static_cast<float *>(normalized_input),
      static_cast<float *>(scaled_gradient), static_cast<float *>(variance),
      static_cast<float *>(summed_scale),
      static_cast<float *>(summed_scaled_input), batch_size, num_features);
}

} // namespace smollnet
