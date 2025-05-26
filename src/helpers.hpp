#pragma once

#include <cstdlib>
#include <cassert>

namespace smollnet {
#define ASSERT(expr, message)                                                  \
  {                                                                            \
    if (!(expr)) {                                                             \
      fprintf(stderr, "Assertion fail at %s:%d: %s\n", __FILE__, __LINE__,     \
              message);                                                        \
      assert(false);                                                           \
    }                                                                          \
  }

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    ASSERT((err == cudaSuccess), cudaGetErrorString(err));                     \
  }

} // namespace smollnet
