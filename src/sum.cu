#include "helpers.hpp"
#include "kernels.cuh"

namespace smollnet {

template <size_t BLOCK_DIM = 256, int32_t MAJOR = ROW_MAJOR>
__global__ void
warp_level_sum(const float *__restrict__ in, float *__restrict__ out,
               size_t dim_size, int64_t s0_in, int64_t s1_in, int64_t s2_in,
               int64_t s0_out, int64_t s1_out, int64_t s2_out, size_t n) {
  // We always launch this kernel as 1D block
  // the only difference can be a grid dim
  int64_t col;
  int64_t row;
  int64_t depth = blockIdx.z;

  int64_t idx;
  int64_t out_idx;
  bool in_bounds;

  if constexpr (MAJOR == ROW_MAJOR) {
    col = blockIdx.x * blockDim.x + threadIdx.x;
    row = blockIdx.y;

    idx = depth * s0_in + row * s1_in + col;
    out_idx = depth * s0_out + row * s1_out;
    in_bounds = (idx < n and col < dim_size);
  } else if (MAJOR == COL_MAJOR) {
    col = blockIdx.y;
    row = blockIdx.x * blockDim.x + threadIdx.x;

    idx = depth * s0_in + row + col * s2_in;
    out_idx = depth * s0_out + col * s2_out;
    in_bounds = (idx < n and row < dim_size);
  } else {
    depth = blockIdx.x * blockDim.x + threadIdx.x;
    row = blockIdx.y;
    col = blockIdx.z;

    idx = depth + row * s1_in + col * s2_in;
    out_idx = row * s1_out + col * s2_out;
    in_bounds = (idx < n and depth < dim_size);
  }

  float v = in_bounds ? in[idx] : 0.0f;

  __shared__ float sMem[BLOCK_DIM / 32];

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);

  if ((threadIdx.x & 31) == 0) {
    sMem[threadIdx.x / 32] = v;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    float acc = 0.0f;
    for (int i = 0; i < BLOCK_DIM / 32; ++i) {
      acc += sMem[i];
    }

    atomicAdd(out + out_idx, acc);
  }
}

template <int32_t VEC_LEN, int32_t MAJOR = ROW_MAJOR>
__global__ void strided_sum_2d(const float *__restrict__ in,
                               float *__restrict__ out, int64_t dim_len,
                               int64_t s0, int64_t s1, int64_t outer_dim_len) {

  int64_t col;
  int64_t row;
  int64_t base_idx;
  int64_t outer_idx;
  int64_t elem_stride;
  int64_t base_vec_idx;

  if constexpr (MAJOR == ROW_MAJOR) {
    col = blockIdx.x * blockDim.x + threadIdx.x;
    row = blockIdx.y;

    outer_idx = col;
    base_idx = row * s0 * VEC_LEN + col * s1;

    elem_stride = s0;
    base_vec_idx = blockIdx.y * VEC_LEN;
  } else {
    col = blockIdx.y;
    row = blockIdx.x * blockDim.x + threadIdx.x;
    outer_idx = row;

    base_idx = row * s0 + col * s1 * VEC_LEN;

    elem_stride = s1;
    base_vec_idx = blockIdx.x * VEC_LEN;
  }

  if (outer_idx >= outer_dim_len)
    return;

  float acc = 0.0f;

#pragma unroll
  for (int elem = 0; elem < VEC_LEN; elem++) {
    const int64_t my_idx = base_idx + elem * elem_stride;
    const float my_val = (base_vec_idx + elem) < dim_len ? in[my_idx] : 0.0f;
    acc += my_val;
  }

  atomicAdd(out + outer_idx, acc);
}

template <int32_t VEC_LEN, int32_t AXIS>
__global__ void strided_sum_3d(const float *__restrict__ in,
                               float *__restrict__ out, int64_t dim_len,
                               int64_t s0, int64_t s1, int64_t s2,
                               int64_t main_axis_max_len) {

  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t row = blockIdx.y;

  int64_t main_axis = col;
  int64_t seconday_axis;
  int64_t base_idx;
  int64_t out_idx;

  int64_t stride;

  if constexpr (AXIS == 0) {
    seconday_axis = blockIdx.z;

    base_idx = row * s1 + col * s2 + blockIdx.z * VEC_LEN * s0;
    stride = s0;

    out_idx = row * main_axis_max_len + col;
  } else if constexpr (AXIS == 1) {
    seconday_axis = row;

    base_idx = row * s1 * VEC_LEN + col * s2 + blockIdx.z * s0;
    stride = s1;

    out_idx = blockIdx.z * main_axis_max_len + col;
  } else {
    row = blockIdx.x * blockDim.x + threadIdx.x;
    col = blockIdx.y;

    main_axis = row;
    seconday_axis = col;

    base_idx = row * s1 + col * s2 * VEC_LEN + blockIdx.z * s0;
    stride = s2;

    out_idx = blockIdx.z * main_axis_max_len + row;
  }

  if (main_axis >= main_axis_max_len)
    return;

  float acc = 0.0f;

#pragma unroll
  for (int elem = 0; elem < VEC_LEN; elem++) {
    const int64_t my_idx = base_idx + elem * stride;
    const float my_val =
        (seconday_axis * VEC_LEN + elem) < dim_len ? in[my_idx] : 0.0f;
    acc += my_val;
  }

  atomicAdd(out + out_idx, acc);
}

void launch_sum_dim0(void *out, void *in, const StrideAndSize &s_input,
                     const StrideAndSize &s_output) {

  const auto d0 = s_input.size[0];
  const auto d1 = s_input.size[1];
  const auto d2 = s_input.size[2];

  const int64_t dim_len = d0;
  constexpr size_t BLOCK = 256;

  // Contigious memory access -> warp level reduce!
  if (s_input.stride[0] == 1) {
    const auto total = d0 * d1 * d2;

    if (s_input.rank == 1) {
      dim3 grid((BLOCK + d0 - 1) / BLOCK, 1, 1);

      warp_level_sum<BLOCK><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[0], s_input.stride[0],
          s_output.stride[0], s_output.stride[0], s_output.stride[0], total);

    } else if (s_input.rank == 2) {
      dim3 grid((BLOCK + d0 - 1) / BLOCK, d1, 1);

      warp_level_sum<BLOCK, COL_MAJOR><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[0], s_input.stride[1],
          s_output.stride[0], s_output.stride[0], s_output.stride[1], total);
    } else {
      dim3 grid((BLOCK + d0 - 1) / BLOCK, d1, d2);

      warp_level_sum<BLOCK, DEPTH_MAJOR><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[1], s_input.stride[2],
          s_output.stride[0], s_output.stride[1], s_output.stride[2], total);
    }
  } else {
    constexpr int32_t VEC_LEN = 2;

    dim3 grid;
    if (s_input.rank == 2) {
      grid = dim3((BLOCK + d1 - 1) / BLOCK, (VEC_LEN + d0 - 1) / VEC_LEN, 1);

      strided_sum_2d<VEC_LEN><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[1], d1);
    } else {
      grid = dim3((BLOCK + d2 - 1) / BLOCK, d1, (VEC_LEN + d0 - 1) / VEC_LEN);

      strided_sum_3d<VEC_LEN, 0><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[1], s_input.stride[2], d2);
    }
  }

  CHECK_CUDA(cudaGetLastError());
}

void launch_sum_dim1(void *out, void *in, const StrideAndSize &s_input,
                     const StrideAndSize &s_output) {
  const auto d0 = s_input.size[0];
  const auto d1 = s_input.size[1];
  const auto d2 = s_input.size[2];

  const int64_t dim_len = d1;
  constexpr size_t BLOCK = 256;

  // Contigious memory access -> warp level reduce!
  if (s_input.stride[1] == 1) {
    const auto total = d0 * d1 * d2;

    if (s_input.rank == 2) {
      dim3 grid((BLOCK + d1 - 1) / BLOCK, d0, 1);

      warp_level_sum<BLOCK><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[0], s_input.stride[1],
          s_output.stride[0], s_output.stride[0], s_output.stride[1], total);
    } else {
      // Transposed
      dim3 grid((BLOCK + d1 - 1) / BLOCK, d2, d0);

      warp_level_sum<BLOCK, COL_MAJOR><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[1], s_input.stride[2],
          s_output.stride[0], s_output.stride[1], s_output.stride[2], total);
    }

  } else {
    constexpr int32_t VEC_LEN = 64;

    if (s_input.rank == 2) {
      dim3 grid((BLOCK + d0 - 1) / BLOCK, (VEC_LEN + d1 - 1) / VEC_LEN, 1);

      strided_sum_2d<VEC_LEN, COL_MAJOR><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[1], d0);
    } else {
      dim3 grid((BLOCK + d2 - 1) / BLOCK, (VEC_LEN + d1 - 1) / VEC_LEN, d0);

      strided_sum_3d<VEC_LEN, 1><<<grid, BLOCK>>>(
          static_cast<const float *>(in), static_cast<float *>(out), dim_len,
          s_input.stride[0], s_input.stride[1], s_input.stride[2], d2);
    }
  }

  CHECK_CUDA(cudaGetLastError());
}

void launch_sum_dim2(void *out, void *in, const StrideAndSize &s_input,
                     const StrideAndSize &s_output) {

  const auto d0 = s_input.size[0];
  const auto d1 = s_input.size[1];
  const auto d2 = s_input.size[2];

  constexpr size_t BLOCK = 256;

  if (s_input.stride[2] == 1) {
    const auto total = d0 * d1 * d2;
    dim3 grid((BLOCK + d2 - 1) / BLOCK, d1, d0);

    warp_level_sum<BLOCK><<<grid, BLOCK>>>(
        static_cast<const float *>(in), static_cast<float *>(out), d2,
        s_input.stride[0], s_input.stride[1], s_input.stride[2],
        s_output.stride[0], s_output.stride[1], s_output.stride[2], total);
  } else {
    constexpr int32_t VEC_LEN = 64;

    dim3 grid((d2 + BLOCK - 1) / BLOCK, (d1 + VEC_LEN - 1) / VEC_LEN, d0);

    strided_sum_3d<VEC_LEN, 2><<<grid, BLOCK>>>(
        static_cast<const float *>(in), static_cast<float *>(out), d2,
        s_input.stride[0], s_input.stride[1], s_input.stride[2], d1);
  }

  CHECK_CUDA(cudaGetLastError());
}

} // namespace smollnet
