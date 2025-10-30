#include "helpers.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace smollnet {

__global__ void init_rng_states_kernel(curandStatePhilox4_32_10_t *states,
                                       std::size_t total,
                                       unsigned long long seed) {
  std::size_t idx =
      threadIdx.x + static_cast<std::size_t>(blockDim.x) * blockIdx.x;

  if (idx >= total)
    return

        curand_init(seed, idx, 0, &states[idx]);
}

__global__ void random_fill_kernel(float *out,
                                   curandStatePhilox4_32_10_t *states,
                                   std::size_t total) {
  std::size_t idx =
      threadIdx.x + static_cast<std::size_t>(blockDim.x) * blockIdx.x;

  if (idx >= total)
    return;

  curandStatePhilox4_32_10_t state = states[idx];

  out[idx] = curand_uniform(&state);

  states[idx] = state;
}

void launch_init_rng_states(curandStatePhilox4_32_10_t *states,
                            std::size_t total, unsigned long long seed) {
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);

  init_rng_states_kernel<<<grid, block>>>(states, total, seed);

  CHECK_CUDA(cudaGetLastError());
}

void launch_random_fill(float *out, curandStatePhilox4_32_10_t *states,
                        std::size_t total) {
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);

  random_fill_kernel<<<grid, block>>>(out, states, total);

  CHECK_CUDA(cudaGetLastError());
}

void launch_random_init(void *out, size_t total) {
  constexpr unsigned long long seed = 1234ULL;
  curandStatePhilox4_32_10_t *d_states = nullptr;

  CHECK_CUDA(cudaMalloc(&d_states, total * sizeof(curandStatePhilox4_32_10_t)));

  launch_init_rng_states(d_states, total, seed);

  CHECK_CUDA(cudaDeviceSynchronize());

  launch_random_fill(static_cast<float*>(out), d_states, total);

  CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace smollnet