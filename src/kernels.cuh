#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "types.hpp"

namespace smollnet {

constexpr int32_t ROW_MAJOR = 0;
constexpr int32_t COL_MAJOR = 1;
constexpr int32_t DEPTH_MAJOR = 2;

struct StrideAndSize {
  int64_t stride[kMaxTensorDims] = {};

  int64_t rank;
  int64_t size[kMaxTensorDims] = {};
};

struct StrideInfo {
  // size of the output operation
  int64_t output_size[kMaxTensorDims] = {};

  int64_t a_stride[kMaxTensorDims] = {};
  int64_t b_stride[kMaxTensorDims] = {};
  int64_t rank;
};

struct SizeInfo {
  int64_t a_size[kMaxTensorDims] = {};
  int64_t b_size[kMaxTensorDims] = {};
};

enum class WelfordType : uint8_t {
  Mean,
  PopulationVariance,
  SampleVariance
};

void launch_fill(void *ptr, DataType dtype, size_t numElems, float val);
void launch_random_init(unsigned long long seed);
void launch_random_fill(void *out, DataType dtype, size_t total);
void launch_negative(void *ptr, DataType dtype, size_t total);

// Binary OPS
void launch_add(void *out, const void *a, const void *b, DataType dtype,
                size_t numElems);
void launch_add_strided(void *dst, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total);
void launch_sub(void *out, const void *a, const void *b, DataType dtype,
                size_t numElems);
void launch_sub_strided(void *out, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total);
void launch_mul(void *out, const void *a, const void *b, DataType dtype,
                size_t numElems);
void launch_mul_strided(void *dst, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total);
void launch_div(void *out, const void *a, const void *b, DataType dtype,
                size_t numElems);
void launch_div_strided(void *dst, const void *a, const void *b,
                        DataType dtype, const StrideInfo &s, size_t total);

void launch_sum_dim(void *out, const void *in, DataType input_dtype,
                    const StrideAndSize &s_input,
                    const StrideAndSize &s_output, int64_t dim);

void launch_matmul(void *out, const void *left, const void *right, DataType dtype,
                   const StrideInfo &strides, const SizeInfo &sizes,
                   size_t total);

// ACTIVATIONS
void launch_relu(void *out, const void *in, DataType dtype, size_t total);
void launch_relu_grad(void *out, const void *grad_out, const void *in,
                      DataType dtype, size_t total);

void launch_gelu(void *out, const void *in, DataType dtype, size_t total);
void launch_gelu_grad(void *out, const void *grad_out, const void *in,
                      DataType dtype, size_t total);

void launch_tanh(void *out, const void *in, DataType dtype, size_t total);
void launch_tanh_grad(void *out, const void *grad_out, const void *in,
                      DataType dtype, size_t total);

void launch_sigmoid(void *out, const void *in, DataType dtype, size_t total);
void launch_sigmoid_grad(void *out, const void *grad_out, const void *in,
                         DataType dtype, size_t total);

void launch_mse(void *out, DataType out_dtype, const void *pred,
                const void *target, DataType input_dtype, size_t total);
void launch_sgd_update(void *p, const void *g, DataType param_dtype,
                       DataType grad_dtype, float lr, size_t total);
void launch_mse_grad(void *grad, const void *pred, const void *target,
                     DataType dtype, float coeff, size_t total);

// NORM
void launch_mean_2d(void *out, DataType out_dtype, const void *in,
                    DataType in_dtype, size_t d0, size_t d1);

void launch_layer_norm(void *out, const void *features, const void *mean,
                       const void *variance, const void *gamma,
                       const void *beta, DataType data_dtype,
                       DataType stats_dtype, size_t batch_size,
                       size_t num_features);

void launch_layer_norm_grad(void *out, const void *normalized_input,
                            const void *scaled_gradient, const void *variance,
                            const void *summed_scale,
                            const void *summed_scaled_input,
                            DataType data_dtype, DataType stats_dtype,
                            size_t batch_size, size_t num_features);

void launch_welford(const void *in, DataType in_dtype, void *out,
                    DataType out_dtype, size_t dim1_len, size_t dim0_len,
                    int32_t dim, WelfordType type);
} // namespace smollnet
