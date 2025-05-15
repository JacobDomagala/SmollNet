#pragma once

namespace smollnet {

void launch_fill(float *ptr, size_t numElems, float val);

// Binary OPS
void launch_add(float *out, float *left, float *right, size_t numElems);
void launch_sub(float *out, float *left, float *right, size_t numElems);

void launch_sum_dim0(void* out, void* in, int64_t d0, int64_t rest);
void launch_sum_dim1(void* out, void* in, int64_t d0, int64_t d1, int64_t d2);

} // namespace smollnet
