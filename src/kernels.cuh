#pragma once

namespace smollnet {

void launch_fill(float *ptr, size_t numElems, float val);
void launch_add(float *out, float *left, float *right, size_t numElems);

} // namespace smollnet
