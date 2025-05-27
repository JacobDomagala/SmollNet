#pragma once

#include <fmt/core.h>
#include <cassert>
#include <cstdlib>

namespace smollnet {
#define ASSERT(expr, message)                                                  \
  {                                                                            \
    if (!(expr)) {                                                             \
      fmt::print("Assertion fail at {}:{}: {}\n", __FILE__, __LINE__,          \
                 message);                                                     \
      assert(false);                                                           \
    }                                                                          \
  }

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    ASSERT((err == cudaSuccess), cudaGetErrorString(err));                     \
  }

} // namespace smollnet
