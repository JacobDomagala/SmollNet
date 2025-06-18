#pragma once

#include <fmt/core.h>
#include <cstdlib>

namespace smollnet {
#define ASSERT(expr, message)                                                  \
  {                                                                            \
    if (!(expr)) {                                                             \
      fmt::print("Assertion fail at {}:{}: {}\n", __FILE__, __LINE__,          \
                 message);                                                     \
      exit(-1);                                                           \
    }                                                                          \
  }

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    ASSERT((err == cudaSuccess), cudaGetErrorString(err));                     \
  }

} // namespace smollnet
