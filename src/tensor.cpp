#include "tensor.hpp"

#include <cuda_runtime.h>

namespace smollnet {

Storage::~Storage() {
  if (--refcount == 0)
    cudaFree(ptr);
}

} // namespace smollnet
