#pragma once

#include <cstdlib>

namespace smollnet {
#define ASSERT(expr, message)                                                  \
  {                                                                            \
    if (!(expr)) {                                                             \
      fprintf(stderr, "Assertion fail at %s:%d: %s\n", __FILE__, __LINE__,     \
              message);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    ASSERT((err == cudaSuccess), cudaGetErrorString(err));                     \
  }

} // namespace smollnet
