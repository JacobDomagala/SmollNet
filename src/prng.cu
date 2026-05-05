#include "helpers.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace smollnet {

namespace {

constexpr int kBlockSize = 256;
constexpr int kBlocksPerSm = 4;

curandStatePhilox4_32_10_t *d_states = nullptr;
std::size_t g_num_states = 0;

} // namespace

__global__ void random_fill_kernel(float *out,
                                   curandStatePhilox4_32_10_t *states,
                                   std::size_t total,
                                   std::size_t num_states) {
  const std::size_t tid =
      threadIdx.x + static_cast<std::size_t>(blockDim.x) * blockIdx.x;

  if (tid >= num_states) {
    return;
  }

  std::size_t idx = tid;
  const std::size_t stride = num_states;

  curandStatePhilox4_32_10_t local_state = states[tid];

  while (idx < total) {
    const float4 r = curand_uniform4(&local_state);

    out[idx] = r.x;
    idx += stride;

    if (idx >= total) break;
    out[idx] = r.y;
    idx += stride;

    if (idx >= total) break;
    out[idx] = r.z;
    idx += stride;

    if (idx >= total) break;
    out[idx] = r.w;
    idx += stride;
  }

  states[tid] = local_state;
}

__global__ void init_rng_states_kernel(curandStatePhilox4_32_10_t *states,
                                       std::size_t num_states,
                                       unsigned long long seed) {
  const std::size_t tid =
      threadIdx.x + static_cast<std::size_t>(blockDim.x) * blockIdx.x;

  if (tid >= num_states) {
    return;
  }

  curand_init(seed, static_cast<unsigned long long>(tid), 0ULL, &states[tid]);
}

std::size_t choose_num_rng_states() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp props{};
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));

  return static_cast<std::size_t>(props.multiProcessorCount) * kBlocksPerSm *
         kBlockSize;
}

void launch_random_init(unsigned long long seed) {
  if (d_states != nullptr) {
    CHECK_CUDA(cudaFree(d_states));
    d_states = nullptr;
    g_num_states = 0;
  }

  g_num_states = choose_num_rng_states();

  CHECK_CUDA(
      cudaMalloc(&d_states, g_num_states * sizeof(curandStatePhilox4_32_10_t)));

  dim3 block_size(kBlockSize);
  dim3 grid_size((g_num_states + block_size.x - 1) / block_size.x);

  init_rng_states_kernel<<<grid_size, block_size>>>(d_states, g_num_states,
                                                    seed);

  CHECK_CUDA(cudaGetLastError());
}

void launch_random_fill(void *out, std::size_t total) {
  if (d_states == nullptr || g_num_states == 0) {
    launch_random_init(1234ULL);
  }

  dim3 block_size(kBlockSize);
  dim3 grid_size((g_num_states + block_size.x - 1) / block_size.x);

  random_fill_kernel<<<grid_size, block_size>>>(static_cast<float *>(out),
                                                d_states, total, g_num_states);

  CHECK_CUDA(cudaGetLastError());
}

} // namespace smollnet