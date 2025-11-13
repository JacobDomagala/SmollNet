#include "dtype_utils.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace smollnet {

namespace {

template <typename T>
__global__ void matmul_kernel(T *__restrict__ C, const T *__restrict__ A,
                              const T *__restrict__ B,
                              const StrideInfo strides, const SizeInfo sizes,
                              const int tile_width) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  const int M = strides.output_size[0];
  const int N = strides.output_size[1];
  const int K = sizes.a_size[1];

  const bool in_bounds = (row < M) && (col < N);

  extern __shared__ float s_mem[];
  float *As = s_mem;
  float *Bs = s_mem + tile_width * tile_width;

  float acc = 0.0f;
  const int num_tiles = (K + tile_width - 1) / tile_width;

  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * tile_width + threadIdx.x;
    const int b_row = t * tile_width + threadIdx.y;

    const int64_t a_offset =
        row * strides.a_stride[0] + a_col * strides.a_stride[1];
    As[threadIdx.y * tile_width + threadIdx.x] =
        (row < M && a_col < K) ? scalar_to_float(A[a_offset]) : 0.0f;

    const int64_t b_offset =
        b_row * strides.b_stride[0] + col * strides.b_stride[1];
    Bs[threadIdx.y * tile_width + threadIdx.x] =
        (b_row < K && col < N) ? scalar_to_float(B[b_offset]) : 0.0f;

    __syncthreads();

    const int elems = min(tile_width, K - t * tile_width);
#pragma unroll
    for (int e = 0; e < elems; ++e) {
      acc +=
          As[threadIdx.y * tile_width + e] * Bs[e * tile_width + threadIdx.x];
    }

    __syncthreads();
  }

  if (in_bounds) {
    C[row * N + col] = scalar_from_float<T>(acc);
  }
}

template <typename T, typename Op>
__global__ void unary_kernel(T *out, const T *in, size_t total, Op op) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total) {
    out[idx] = scalar_from_float<T>(op(scalar_to_float(in[idx])));
  }
}

struct ReluOp {
  __device__ float operator()(float x) const { return fmaxf(x, 0.0f); }
};

struct GeluOp {
  __device__ float operator()(float x) const {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x *
           (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
  }
};

struct TanhOp {
  __device__ float operator()(float x) const { return tanhf(x); }
};

struct SigmoidOp {
  __device__ float operator()(float x) const {
    return 1.0f / (1.0f + expf(-x));
  }
};

template <typename T>
__global__ void relu_grad_kernel(T *out, const T *grad_out, const T *in,
                                 size_t total) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total) {
    const float input = scalar_to_float(in[idx]);
    const float grad = input > 0.0f ? scalar_to_float(grad_out[idx]) : 0.0f;
    out[idx] = scalar_from_float<T>(grad);
  }
}

template <typename T>
__global__ void gelu_grad_kernel(T *out, const T *grad_out, const T *in,
                                 size_t total) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total) {
    constexpr float a = 0.7978845608f;
    constexpr float b = 0.044715f;
    const float x = scalar_to_float(in[idx]);
    const float x3 = x * x * x;
    const float h = x + b * x3;
    const float tanh_ax = tanhf(a * h);
    const float sech2 = 1.0f - tanh_ax * tanh_ax;
    const float h_prime = 1.0f + 3.0f * b * x * x;

    const float g =
        0.5f * (1.0f + tanh_ax) + 0.5f * x * sech2 * a * h_prime;
    out[idx] = scalar_from_float<T>(scalar_to_float(grad_out[idx]) * g);
  }
}

template <typename T>
__global__ void tanh_grad_kernel(T *out, const T *grad_out, const T *in,
                                 size_t total) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < total) {
    const float x = scalar_to_float(in[idx]);
    const float grad = scalar_to_float(grad_out[idx]) * (1.0f - x * x);
    out[idx] = scalar_from_float<T>(grad);
  }
}

template <typename T>
__global__ void sigmoid_grad_kernel(T *output, const T *grad_out,
                                    const T *input, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const float x = scalar_to_float(input[idx]);
    const float grad = scalar_to_float(grad_out[idx]) * x * (1.0f - x);
    output[idx] = scalar_from_float<T>(grad);
  }
}

template <typename InT>
__global__ void mse_partial_kernel(float *partial,
                                   const InT *__restrict__ pred,
                                   const InT *__restrict__ target,
                                   std::size_t n) {
  __shared__ float sMem[32];

  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;

  float local_sum = 0.0f;

  for (; idx < n; idx += stride) {
    const float diff =
        scalar_to_float(pred[idx]) - scalar_to_float(target[idx]);
    local_sum += diff * diff;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  if (lane == 0) {
    sMem[warp] = local_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;

  if (warp == 0) {
    block_sum = (lane < num_warps) ? sMem[lane] : 0.0f;

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, off);
    }
  }

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = block_sum;
  }
}

template <typename OutT>
__global__ void mse_finalize_kernel(OutT *out, const float *partial,
                                    std::size_t num_partials,
                                    std::size_t n) {
  float acc = 0.0f;
  for (std::size_t idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
    acc += partial[idx];
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }

  __shared__ float sMem[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  if (lane == 0) {
    sMem[warp] = acc;
  }

  __syncthreads();

  if (warp == 0) {
    acc = (lane < num_warps) ? sMem[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, off);
    }
  }

  if (threadIdx.x == 0) {
    out[0] = scalar_from_float<OutT>(acc / static_cast<float>(n));
  }
}

template <typename ParamT, typename GradT>
__global__ void sgd_kernel(ParamT *w, const GradT *grad, float lr,
                           size_t total) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    const float updated =
        scalar_to_float(w[idx]) - lr * scalar_to_float(grad[idx]);
    w[idx] = scalar_from_float<ParamT>(updated);
  }
}

template <typename T>
__global__ void mse_grad_kernel(T *g, const T *p, const T *t, float coeff,
                                size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float grad =
        coeff * (scalar_to_float(p[idx]) - scalar_to_float(t[idx]));
    g[idx] = scalar_from_float<T>(grad);
  }
}

template <typename OutT, typename InT>
__global__ void mean_2d_kernel(OutT *out, const InT *in, size_t d0,
                               size_t d1) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= d0) {
    return;
  }

  float acc = 0.0f;
  for (size_t i = 0; i < d1; ++i) {
    acc += scalar_to_float(in[idx * d1 + i]);
  }

  out[idx] = scalar_from_float<OutT>(acc / static_cast<float>(d1));
}

template <typename DataT, typename StatsT>
__global__ void layer_norm_kernel(DataT *out, const DataT *features,
                                  const StatsT *mean, const StatsT *variance,
                                  const DataT *gamma, const DataT *beta,
                                  size_t batch_size, size_t num_features) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto total = batch_size * num_features;

  if (idx >= total) {
    return;
  }

  const size_t batch_num = idx / num_features;

  constexpr float epsilon = 1e-5f;
  const float normalized =
      (scalar_to_float(features[idx]) - scalar_to_float(mean[batch_num])) /
      sqrtf(scalar_to_float(variance[batch_num]) + epsilon);

  const float value = scalar_to_float(gamma[batch_num]) * normalized +
                      scalar_to_float(beta[batch_num]);
  out[idx] = scalar_from_float<DataT>(value);
}

template <typename DataT, typename StatsT>
__global__ void layer_norm_grad_kernel(
    DataT *out_grad, const DataT *normalized_input,
    const DataT *scaled_gradient, const StatsT *variance,
    const StatsT *summed_scale, const StatsT *summed_scaled_input,
    size_t batch_size, size_t num_features) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t total = batch_size * num_features;
  if (idx >= total) {
    return;
  }

  const size_t row = idx / num_features;

  constexpr float eps = 1e-5f;
  const float inv_std = rsqrtf(scalar_to_float(variance[row]) + eps);
  const float m1 = scalar_to_float(summed_scale[row]) / num_features;
  const float m2 = scalar_to_float(summed_scaled_input[row]) / num_features;

  const float hat_x = scalar_to_float(normalized_input[idx]);
  const float delta = scalar_to_float(scaled_gradient[idx]);

  const float res = inv_std * (delta - m1 - hat_x * m2);
  out_grad[idx] = scalar_from_float<DataT>(res);
}

template <typename Op>
void launch_unary(void *out, const void *in, DataType dtype, size_t total,
                  Op op) {
  const int block = 256;
  const int grid = (total + block - 1) / block;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    unary_kernel<T, Op><<<grid, block>>>(static_cast<T *>(out),
                                         static_cast<const T *>(in), total,
                                         op);
  });

  CHECK_CUDA(cudaGetLastError());
}

} // namespace

void launch_matmul(void *out, const void *left, const void *right,
                   DataType dtype, const StrideInfo &strides,
                   const SizeInfo &sizes, size_t /*total*/) {
  constexpr int TILE = 16;
  dim3 block(TILE, TILE);

  const int M = strides.output_size[0];
  const int N = strides.output_size[1];

  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  const size_t smem_bytes = 2 * TILE * TILE * sizeof(float);

  dispatch_float_dtype(dtype, [&]<typename T>() {
    matmul_kernel<T><<<grid, block, smem_bytes>>>(
        static_cast<T *>(out), static_cast<const T *>(left),
        static_cast<const T *>(right), strides, sizes, TILE);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_relu(void *out, const void *in, DataType dtype, size_t total) {
  launch_unary(out, in, dtype, total, ReluOp{});
}

void launch_relu_grad(void *out, const void *grad_out, const void *in,
                      DataType dtype, size_t total) {
  const int block = 256;
  const int grid = (total + block - 1) / block;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    relu_grad_kernel<T><<<grid, block>>>(
        static_cast<T *>(out), static_cast<const T *>(grad_out),
        static_cast<const T *>(in), total);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_gelu(void *out, const void *in, DataType dtype, size_t total) {
  launch_unary(out, in, dtype, total, GeluOp{});
}

void launch_gelu_grad(void *out, const void *grad_out, const void *in,
                      DataType dtype, size_t total) {
  const int block = 256;
  const int grid = (total + block - 1) / block;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    gelu_grad_kernel<T><<<grid, block>>>(
        static_cast<T *>(out), static_cast<const T *>(grad_out),
        static_cast<const T *>(in), total);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_tanh(void *out, const void *in, DataType dtype, size_t total) {
  launch_unary(out, in, dtype, total, TanhOp{});
}

void launch_tanh_grad(void *out, const void *grad_out, const void *in,
                      DataType dtype, size_t total) {
  const int block = 256;
  const int grid = (total + block - 1) / block;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    tanh_grad_kernel<T><<<grid, block>>>(
        static_cast<T *>(out), static_cast<const T *>(grad_out),
        static_cast<const T *>(in), total);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_sigmoid(void *out, const void *in, DataType dtype, size_t total) {
  launch_unary(out, in, dtype, total, SigmoidOp{});
}

void launch_sigmoid_grad(void *out, const void *grad_out, const void *in,
                         DataType dtype, size_t total) {
  const int block = 256;
  const int grid = (total + block - 1) / block;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    sigmoid_grad_kernel<T><<<grid, block>>>(
        static_cast<T *>(out), static_cast<const T *>(grad_out),
        static_cast<const T *>(in), static_cast<int>(total));
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_mse(void *out, DataType out_dtype, const void *pred,
                const void *target, DataType input_dtype, size_t total) {
  constexpr int BLOCK_SIZE = 256;
  const int grid = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

  float *partials = nullptr;
  CHECK_CUDA(cudaMalloc(&partials, grid * sizeof(float)));

  dispatch_float_dtype(input_dtype, [&]<typename InT>() {
    mse_partial_kernel<InT><<<grid, BLOCK_SIZE>>>(
        partials, static_cast<const InT *>(pred),
        static_cast<const InT *>(target), total);
  });
  CHECK_CUDA(cudaGetLastError());

  dispatch_float_dtype(out_dtype, [&]<typename OutT>() {
    mse_finalize_kernel<OutT><<<1, BLOCK_SIZE>>>(
        static_cast<OutT *>(out), partials, static_cast<std::size_t>(grid),
        total);
  });
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaFree(partials));
}

void launch_sgd_update(void *p, const void *g, DataType param_dtype,
                       DataType grad_dtype, float lr, size_t total) {
  dim3 block = 256;
  dim3 grid = (total + block.x - 1) / block.x;

  dispatch_float_dtype(param_dtype, [&]<typename ParamT>() {
    dispatch_float_dtype(grad_dtype, [&]<typename GradT>() {
      sgd_kernel<ParamT, GradT><<<grid, block>>>(
          static_cast<ParamT *>(p), static_cast<const GradT *>(g), lr, total);
    });
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_mse_grad(void *grad, const void *pred, const void *target,
                     DataType dtype, float coeff, size_t total) {
  const int block = 256;
  const int grid = (total + block - 1) / block;

  dispatch_float_dtype(dtype, [&]<typename T>() {
    mse_grad_kernel<T><<<grid, block>>>(
        static_cast<T *>(grad), static_cast<const T *>(pred),
        static_cast<const T *>(target), coeff, total);
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_mean_2d(void *out, DataType out_dtype, const void *in,
                    DataType in_dtype, size_t d0, size_t d1) {
  dim3 block = 256;
  dim3 grid = (block.x + d0 - 1) / block.x;

  dispatch_float_dtype(out_dtype, [&]<typename OutT>() {
    dispatch_float_dtype(in_dtype, [&]<typename InT>() {
      mean_2d_kernel<OutT, InT><<<grid, block>>>(
          static_cast<OutT *>(out), static_cast<const InT *>(in), d0, d1);
    });
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_layer_norm(void *out, const void *features, const void *mean,
                       const void *variance, const void *gamma,
                       const void *beta, DataType data_dtype,
                       DataType stats_dtype, size_t batch_size,
                       size_t num_features) {
  dim3 block = 256;
  const size_t total = batch_size * num_features;
  dim3 grid = (block.x + total - 1) / block.x;

  dispatch_float_dtype(data_dtype, [&]<typename DataT>() {
    dispatch_float_dtype(stats_dtype, [&]<typename StatsT>() {
      layer_norm_kernel<DataT, StatsT><<<grid, block>>>(
          static_cast<DataT *>(out), static_cast<const DataT *>(features),
          static_cast<const StatsT *>(mean),
          static_cast<const StatsT *>(variance),
          static_cast<const DataT *>(gamma), static_cast<const DataT *>(beta),
          batch_size, num_features);
    });
  });

  CHECK_CUDA(cudaGetLastError());
}

void launch_layer_norm_grad(void *out, const void *normalized_input,
                            const void *scaled_gradient, const void *variance,
                            const void *summed_scale,
                            const void *summed_scaled_input,
                            DataType data_dtype, DataType stats_dtype,
                            size_t batch_size, size_t num_features) {
  dim3 block = 256;
  const size_t total = batch_size * num_features;
  dim3 grid = (block.x + total - 1) / block.x;

  dispatch_float_dtype(data_dtype, [&]<typename DataT>() {
    dispatch_float_dtype(stats_dtype, [&]<typename StatsT>() {
      layer_norm_grad_kernel<DataT, StatsT><<<grid, block>>>(
          static_cast<DataT *>(out),
          static_cast<const DataT *>(normalized_input),
          static_cast<const DataT *>(scaled_gradient),
          static_cast<const StatsT *>(variance),
          static_cast<const StatsT *>(summed_scale),
          static_cast<const StatsT *>(summed_scaled_input), batch_size,
          num_features);
    });
  });

  CHECK_CUDA(cudaGetLastError());
}

} // namespace smollnet
