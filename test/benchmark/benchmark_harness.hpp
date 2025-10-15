#pragma once

#include "helpers.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>

namespace smollnet::bench {

constexpr int kMinimumWarmupIterations = 10;
// Fallback floor when L2 size is unavailable or very small.
constexpr size_t kDefaultL2FlushBytes = 64ull * 1024ull * 1024ull;

struct RunConfig {
  int iterations = 100;
  int warmup = 10;
  size_t l2_flush_bytes = 0;

  int effective_warmup() const noexcept {
    // A minimum warmup helps clocks/P-states settle before timing.
    return std::max(warmup, kMinimumWarmupIterations);
  }
};

struct TimingStats {
  double avg_ms;
  double min_ms;
  double max_ms;
};

namespace ansi {
constexpr std::string_view kReset = "\033[0m";
constexpr std::string_view kBoldCyan = "\033[1;36m";
constexpr std::string_view kBoldBlue = "\033[1;34m";
constexpr std::string_view kBoldGreen = "\033[1;32m";
constexpr std::string_view kBoldYellow = "\033[1;33m";
constexpr std::string_view kBoldMagenta = "\033[1;35m";
constexpr std::string_view kBoldRed = "\033[1;31m";
constexpr std::string_view kDim = "\033[2m";
constexpr std::string_view kWhite = "\033[37m";
} // namespace ansi

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t bytes) : size_bytes_(bytes) {
    CHECK_CUDA(cudaMalloc(&ptr_, size_bytes_));
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
    other.ptr_ = nullptr;
    other.size_bytes_ = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }

    ptr_ = other.ptr_;
    size_bytes_ = other.size_bytes_;
    other.ptr_ = nullptr;
    other.size_bytes_ = 0;
    return *this;
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  void *data() const noexcept { return ptr_; }
  size_t size_bytes() const noexcept { return size_bytes_; }

private:
  void *ptr_ = nullptr;
  size_t size_bytes_ = 0;
};

inline size_t recommended_l2_flush_bytes() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  int l2_cache_bytes = 0;
  CHECK_CUDA(
      cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, device));

  if (l2_cache_bytes <= 0) {
    return kDefaultL2FlushBytes;
  }

  // 2x L2 is a simple way to evict most data left by the previous iteration.
  return std::max(kDefaultL2FlushBytes,
                  2 * static_cast<size_t>(l2_cache_bytes));
}

inline bool colors_enabled() {
  static const bool enabled = [] {
    const char *no_color = std::getenv("NO_COLOR");
    const char *term = std::getenv("TERM");

    // Keep escape codes out of logs and other non-interactive output.
    return no_color == nullptr && term != nullptr &&
           std::string_view(term) != "dumb" &&
           isatty(fileno(stdout)) != 0;
  }();

  return enabled;
}

inline std::string paint(std::string_view text, std::string_view color) {
  if (!colors_enabled()) {
    return std::string(text);
  }

  return fmt::format("{}{}{}", color, text, ansi::kReset);
}

template <typename... Args>
inline std::string paintf(std::string_view color,
                          fmt::format_string<Args...> fmt_str,
                          Args &&...args) {
  return paint(fmt::format(fmt_str, std::forward<Args>(args)...), color);
}

struct DisplayField {
  std::string label;
  std::string value;
  std::string_view value_color = ansi::kWhite;
};

template <typename... Args>
inline DisplayField field(std::string_view label, std::string_view value_color,
                          fmt::format_string<Args...> fmt_str,
                          Args &&...args) {
  return {std::string(label), fmt::format(fmt_str, std::forward<Args>(args)...),
          value_color};
}

inline void print_fields(std::initializer_list<DisplayField> fields) {
  bool is_first = true;
  for (const auto &cur_field : fields) {
    if (!is_first) {
      fmt::print("  ");
    }
    is_first = false;
    fmt::print("{}={}", paint(cur_field.label, ansi::kDim),
               paint(cur_field.value, cur_field.value_color));
  }
  fmt::print("\n");
}

inline void print_banner(std::string_view benchmark_name,
                         const RunConfig &config) {
  fmt::print("{}\n", paint(benchmark_name, ansi::kBoldCyan));
  print_fields({
      field("iterations", ansi::kBoldYellow, "{}", config.iterations),
      field("warmup", ansi::kBoldYellow, "{}", config.effective_warmup()),
      field("l2_flush_bytes", ansi::kBoldMagenta, "{}", config.l2_flush_bytes),
  });
}

template <typename LaunchFn>
inline TimingStats measure_cuda_operation(const RunConfig &config,
                                          LaunchFn &&launch_op) {
  std::optional<DeviceBuffer> l2_flush_buffer;
  if (config.l2_flush_bytes > 0) {
    l2_flush_buffer.emplace(config.l2_flush_bytes);
  }

  for (int i = 0; i < config.effective_warmup(); ++i) {
    launch_op();
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> latencies_ms;
  latencies_ms.reserve(config.iterations);

  for (int i = 0; i < config.iterations; ++i) {
    if (l2_flush_buffer.has_value()) {
      // Flush before each measured launch to avoid cross-iteration L2 hits.
      CHECK_CUDA(cudaMemsetAsync(l2_flush_buffer->data(), i,
                                 l2_flush_buffer->size_bytes()));
    }

    // CUDA events measure device-side elapsed time instead of host overhead.
    CHECK_CUDA(cudaEventRecord(start));
    launch_op();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    latencies_ms.push_back(elapsed_ms);
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  const auto [min_it, max_it] =
      std::minmax_element(latencies_ms.begin(), latencies_ms.end());
  const double avg_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(),
                                        0.0) /
                        static_cast<double>(latencies_ms.size());

  return {
      avg_ms,
      static_cast<double>(*min_it),
      static_cast<double>(*max_it),
  };
}

} // namespace smollnet::bench
