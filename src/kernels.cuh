#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace smollnet {

constexpr int32_t ROW_MAJOR = 0;
constexpr int32_t COL_MAJOR = 1;
struct StrideAndSize {
  std::array<int64_t, 3> stride;

  int rank;
  std::array<int64_t, 3> size;
};

struct StrideInfo {
  // size of the output operation
  int64_t output_size[3];

  int64_t a_stride[3];
  int64_t b_stride[3];
  int rank;
};

struct SizeInfo {
  int64_t a_size[3];
  int64_t b_size[3];
};

void launch_fill(float *ptr, size_t numElems, float val);
void launch_random_init(void *out, size_t total);
void launch_negative(void *ptr, size_t total);

// Binary OPS
void launch_add(float *out, float *a, float *b, size_t numElems);
void launch_add_strided(void *dst, void *a, void *b, const StrideInfo &s,
                        size_t total);
void launch_sub(float *out, float *a, float *b, size_t numElems);
void launch_sub_strided(void *out, void *a, void *b, const StrideInfo &s,
                        size_t total);
void launch_mul(float *out, float *a, float *b, size_t numElems);
void launch_mul_strided(void *dst, void *a, void *b, const StrideInfo &s,
                        size_t total);

void launch_sum_dim0(void *out, void *in, const StrideAndSize& s_input, const StrideAndSize& s_output);
void launch_sum_dim1(void *out, void *in, const StrideAndSize& s_input, const StrideAndSize& s_output);
void launch_sum_dim2(void *out, void *in, const StrideAndSize& s_input, const StrideAndSize& s_output);

void launch_matmul(void *out, void *left, void *right,
                   const StrideInfo &strides, const SizeInfo &sizes,
                   size_t total);

// ACTIVATIONS
void launch_relu(void *out, void *in, size_t total);
void launch_relu_grad(void *out, void *grad_out, void *in, size_t total);

void launch_gelu(void *out, void *in, size_t total);
void launch_gelu_grad(void *out, void *grad_out, void *in, size_t total);

void launch_tanh(void *out, void *in, size_t total);
void launch_tanh_grad(void *out, void *grad_out, void *in, size_t total);

void launch_sigmoid(void *out, void *in, size_t total);
void launch_sigmoid_grad(void *out, void *grad_out, void *in, size_t total);

void launch_mse(void *out, void *pred, void *target, size_t total);
void launch_sgd_update(void *p, void *g, float lr, size_t total);
void launch_mse_grad(void *grad, void *pred, void *target, float coeff,
                     size_t total);

// NORM
void launch_mean_2d(void *out, void *in, size_t d0, size_t d1);
void launch_variance(void *variance, void *staging_buffer, void *in, void *mean,
                     size_t batch_size, size_t num_features);
void launch_layer_norm(void *out, void *features, void *mean, void *variance,
                       void *gamma, void *beta, size_t batch_size,
                       size_t num_features);

void launch_layer_norm_grad(void *out, void *normalized_input,
                            void *scaled_gradient, void *variance,
                            void *summed_scale, void *summed_scaled_input,
                            size_t batch_size, size_t num_features);
} // namespace smollnet
