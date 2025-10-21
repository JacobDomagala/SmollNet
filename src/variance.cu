#include <cstdio>
#include <cuda.h>

#include "kernels.cuh"
namespace smollnet {

struct WelfordData {
  int count = 0;
  float mean = 0.0f;
  float M2 = 0.0f;
};

__device__ __forceinline__ void update(WelfordData &data, float val,
                                       bool is_valid) {

  if (is_valid) {
    data.count++;
    float delta = val - data.mean;
    data.mean += delta / data.count;
    float delta2 = val - data.mean;
    data.M2 += delta * delta2;
  }
}

__device__ __forceinline__ WelfordData merge(const WelfordData &a,
                                             const WelfordData &b) {
  if (a.count == 0)
    return b;
  if (b.count == 0)
    return a;

  WelfordData out;
  float delta = b.mean - a.mean;
  out.count = a.count + b.count;

  float inv_n = __fdividef(1.0f, (float)out.count);
  out.mean = a.mean + delta * b.count * inv_n;
  out.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count * inv_n;
  return out;
}

template <int32_t CHUNK_SIZE, uint32_t BLOCK_DIM>
__global__ void welford_row_first_pass(const float *__restrict__ in,
                                       WelfordData *__restrict__ out,
                                       const size_t num_features,
                                       const size_t num_rows) {

  int feature = threadIdx.x + blockIdx.x * blockDim.x;
  int batch = blockIdx.y;

  bool is_valid = feature < num_features;

  uint32_t base_idx = feature + batch * num_features * CHUNK_SIZE;

  WelfordData localData;

  __shared__ WelfordData sMem[BLOCK_DIM / 32];

#pragma unroll
  for (int32_t chunk = 0; chunk < CHUNK_SIZE; ++chunk) {

    uint32_t idx = base_idx + num_features * chunk;
    uint32_t row = batch * CHUNK_SIZE + chunk;

    // This will be true for all threads in a block
    if (row >= num_rows)
      return;

    float v = is_valid ? in[idx] : 0.0f;
    update(localData, v, is_valid);

#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
      auto count = __shfl_down_sync(0xffffffff, localData.count, off);
      auto m2 = __shfl_down_sync(0xffffffff, localData.M2, off);
      auto mean = __shfl_down_sync(0xffffffff, localData.mean, off);

      localData = merge(localData, WelfordData{count, mean, m2});
    }

    __syncthreads();

    if (threadIdx.x % 32 == 0) {
      sMem[threadIdx.x / 32] = localData;
    }

    __syncthreads();

    // Final merge
    if (threadIdx.x == 0) {
      localData = {};
#pragma unroll
      for (int32_t i = 0; i < BLOCK_DIM / 32; ++i) {
        localData = merge(localData, sMem[i]);
      }

      out[row * gridDim.x + blockIdx.x] = localData;
    }

    localData = {};
  }
}

template <int32_t BLOCK_DIM>
__global__ void
welford_row_second_pass(const WelfordData *__restrict__ in,
                        float *__restrict__ out, const int32_t num_elems,
                        const int32_t num_iter, WelfordType type) {
  const auto col = threadIdx.x;
  const auto row = blockIdx.x;

  bool is_valid = col < num_elems;

  WelfordData final_result = {};
  __shared__ WelfordData sMem[BLOCK_DIM / 32];

  for (int32_t iter = 0; iter < num_iter; ++iter) {

    const auto new_col = col + iter * BLOCK_DIM;
    const auto idx = new_col + row * num_elems;
    is_valid = new_col < num_elems;

    WelfordData val = is_valid ? in[idx] : WelfordData{};

#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
      auto count = __shfl_down_sync(0xffffffff, val.count, off);
      auto m2 = __shfl_down_sync(0xffffffff, val.M2, off);
      auto mean = __shfl_down_sync(0xffffffff, val.mean, off);

      val = merge(val, WelfordData{count, mean, m2});
    }

    __syncthreads();

    if (threadIdx.x % 32 == 0) {
      sMem[threadIdx.x / 32] = val;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      WelfordData local{};
#pragma unroll
      for (int32_t i = 0; i < BLOCK_DIM / 32; ++i) {
        local = merge(local, sMem[i]);
      }

      final_result = merge(final_result, local);
    }
  }

  if (threadIdx.x == 0) {
    switch (type) {
    case WelfordType::Mean: {
      out[blockIdx.x] = final_result.mean;
    } break;

    case WelfordType::PopulationVariance: {
      out[blockIdx.x] = final_result.M2 / final_result.count;
    } break;

    case WelfordType::SampleVariance: {
      out[blockIdx.x] = final_result.M2 / (final_result.count - 1);
    } break;
    }
  }
}

__global__ void welford_column_first_pass(const float *__restrict__ in,
                                          WelfordData *__restrict__ out,
                                          const size_t num_features,
                                          const size_t size) {

  int feature = threadIdx.x + blockIdx.x * blockDim.x;
  int batch = (threadIdx.y + blockIdx.y * blockDim.y) * 32;

  if (feature >= num_features)
    return;

  WelfordData localData;

  const uint32_t base_idx = batch * num_features + feature;
  const uint32_t output_idx = num_features * blockIdx.y + feature;

#pragma unroll
  for (uint32_t part = 0; part < 32; ++part) {
    const uint32_t offset = part * num_features;
    uint32_t idx = base_idx + offset;

    if (idx >= size) {
      out[output_idx] = localData;
      return;
    }

    float v = in[idx];

    update(localData, v, true);
  }

  out[output_idx] = localData;
}

__global__ void welford_column_second_pass(const WelfordData *__restrict__ in,
                                           float *__restrict__ out,
                                           const size_t num_features,
                                           const uint32_t num_rows,
                                           WelfordType type) {
  const auto base_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (base_idx >= num_features)
    return;

  WelfordData local = in[base_idx];

#pragma unroll
  for (uint32_t row = 1; row < num_rows; ++row) {
    const auto idx = row * num_features + base_idx;

    local = merge(local, in[idx]);
  }

  switch (type) {
  case WelfordType::Mean: {
    out[base_idx] = local.mean;
  } break;

  case WelfordType::PopulationVariance: {
    out[base_idx] = local.M2 / local.count;
  } break;

  case WelfordType::SampleVariance: {
    out[base_idx] = local.M2 / (local.count - 1);
  } break;
  }
}

void launch_welford(void *in, void *out, size_t num_features, size_t batch_size,
                    int32_t dim, WelfordType type) {
  constexpr int32_t CHUNK_SIZE = 4;
  constexpr int32_t BLOCK_DIM = 256;

  dim3 block_size(BLOCK_DIM, 1);
  size_t features_dim = (num_features + BLOCK_DIM - 1) / BLOCK_DIM;
  uint32_t batch_dim = (batch_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
  dim3 grid_size(features_dim, batch_dim);

  void *staging_buffer;
  cudaMalloc(&staging_buffer, sizeof(WelfordData) * features_dim * batch_size);

  if (dim == 0) {
    welford_row_first_pass<CHUNK_SIZE, BLOCK_DIM><<<grid_size, block_size>>>(
        static_cast<const float *>(in),
        static_cast<WelfordData *>(staging_buffer), num_features, batch_size);

    const int32_t num_iter = (features_dim + BLOCK_DIM - 1) / BLOCK_DIM;
    welford_row_second_pass<BLOCK_DIM><<<batch_size, block_size>>>(
        static_cast<const WelfordData *>(staging_buffer),
        static_cast<float *>(out), features_dim, num_iter, type);
  } else if (dim == 1) {
    welford_column_first_pass<<<grid_size, block_size>>>(
        static_cast<const float *>(in),
        static_cast<WelfordData *>(staging_buffer), num_features,
        num_features * batch_size);

    welford_column_second_pass<<<features_dim, block_size>>>(
        static_cast<const WelfordData *>(staging_buffer),
        static_cast<float *>(out), num_features, batch_dim, type);
  }

  cudaFree(staging_buffer);
}

} // namespace smollnet
