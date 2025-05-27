#pragma once

namespace smollnet {

void launch_fill(float *ptr, size_t numElems, float val);
void launch_random_init(void *out, size_t total);

// Binary OPS
void launch_add(float *out, float *left, float *right, size_t numElems);
void launch_sub(float *out, float *left, float *right, size_t numElems);

void launch_sum_dim0(void *out, void *in, int64_t d0, int64_t rest);
void launch_sum_dim1(void *out, void *in, int64_t d0, int64_t d1, int64_t d2);
void launch_sum_dim2(void *out, void *in, int64_t d0, int64_t d1, int64_t d2);

void launch_matmul(void *out, void *left, void *right, int64_t ldims[3],
                   int64_t rdims[3], size_t total);

// ACTIVATIONS
void launch_relu(void *out, void *in, size_t total);
void launch_relu_grad(void *out, void *in, size_t total);

void launch_tanh(void *out, void *in, size_t total);
void launch_tanh_grad(void *out, void *in, size_t total);

void launch_sigmoid(void *out, void *in, size_t total);
void launch_sigmoid_grad(void *out, void *in, size_t total);

} // namespace smollnet
