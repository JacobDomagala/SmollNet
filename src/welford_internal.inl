#pragma once

#include <cstdint>

namespace smollnet::welford_internal {

// Internal kernel tuning knobs shared with the benchmark bandwidth model.
constexpr int32_t kRowChunkSize = 4;
constexpr int32_t kColChunkSize = 32;
constexpr int32_t kBlockDim = 256;

} // namespace smollnet::welford_internal
