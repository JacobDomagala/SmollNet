#include "helpers.hpp"
#include "kernels.cuh"
#include "tensor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <unistd.h>

namespace smollnet {
namespace {

constexpr int kMinimumWarmupIterations = 10;
constexpr size_t kDefaultL2FlushBytes = 64ull * 1024ull * 1024ull;

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

struct BenchmarkCase {
  size_t batch_size;
  size_t num_features;
};

struct BenchmarkConfig {
  BenchmarkCase single_case{8192, 4096};
  int iterations = 100;
  int warmup = 10;
  bool use_default_suite = true;
};

size_t parse_size_arg(const char *text, const char *name) {
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);

  ASSERT(end != nullptr && *end == '\0',
         fmt::format("Invalid {} value '{}'", name, text));
  ASSERT(value > 0, fmt::format("{} must be greater than zero", name));

  return static_cast<size_t>(value);
}

int parse_int_arg(const char *text, const char *name) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);

  ASSERT(end != nullptr && *end == '\0',
         fmt::format("Invalid {} value '{}'", name, text));
  ASSERT(value > 0, fmt::format("{} must be greater than zero", name));

  return static_cast<int>(value);
}

constexpr std::array<BenchmarkCase, 8> kDefaultCases = {{
    {4096, 1024},
    {16384, 1024},
    {65536, 1024},
    {4096, 4096},
    {8192, 4096},
    {16384, 4096},
    {2048, 16384},
    {4096, 16384},
}};

bool colors_enabled() {
  static const bool enabled = [] {
    const char *no_color = std::getenv("NO_COLOR");
    const char *term = std::getenv("TERM");

    return no_color == nullptr && term != nullptr &&
           std::string_view(term) != "dumb" &&
           isatty(fileno(stdout)) != 0;
  }();

  return enabled;
}

std::string paint(std::string_view text, std::string_view color) {
  if (!colors_enabled()) {
    return std::string(text);
  }

  return fmt::format("{}{}{}", color, text, ansi::kReset);
}

template <typename... Args>
std::string paintf(std::string_view color, fmt::format_string<Args...> fmt_str,
                   Args &&...args) {
  return paint(fmt::format(fmt_str, std::forward<Args>(args)...), color);
}

BenchmarkConfig parse_args(int argc, char **argv) {
  BenchmarkConfig cfg;

  if (argc == 1) {
    return cfg;
  }

  ASSERT(argc == 3 || argc == 5,
         "Usage: variance_benchmark "
         "[batch_size num_features [iterations warmup]]");

  cfg.use_default_suite = false;
  cfg.single_case.batch_size = parse_size_arg(argv[1], "batch_size");
  cfg.single_case.num_features = parse_size_arg(argv[2], "num_features");

  if (argc == 5) {
    cfg.iterations = parse_int_arg(argv[3], "iterations");
    cfg.warmup = parse_int_arg(argv[4], "warmup");
  }

  return cfg;
}

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

size_t recommended_l2_flush_bytes() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  int l2_cache_bytes = 0;
  CHECK_CUDA(
      cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, device));

  if (l2_cache_bytes <= 0) {
    return kDefaultL2FlushBytes;
  }

  return std::max(kDefaultL2FlushBytes,
                  2 * static_cast<size_t>(l2_cache_bytes));
}

double bytes_per_iteration(size_t batch_size, size_t num_features) {
  const double total =
      static_cast<double>(batch_size) * static_cast<double>(num_features);

  // launch_variance is a two-pass implementation:
  // 1) read input and mean, then write staging_buffer
  // 2) read staging_buffer, then write variance
  return (4.0 * total + static_cast<double>(batch_size)) * sizeof(float);
}

struct BenchmarkResult {
  double avg_ms;
  double min_ms;
  double max_ms;
  double total_elems;
  double effective_gb_per_sec;
  double bytes_per_iter;
  float sample_variance;
};

BenchmarkResult run_case(const BenchmarkCase &cfg, int iterations, int warmup,
                         size_t l2_flush_bytes) {
  Tensor input =
      rand({static_cast<int64_t>(cfg.batch_size),
            static_cast<int64_t>(cfg.num_features)},
           DataType::f32, Device::CUDA);
  Tensor mean = zeros({static_cast<int64_t>(cfg.batch_size), 1}, DataType::f32,
                      Device::CUDA);
  Tensor variance =
      zeros({static_cast<int64_t>(cfg.batch_size), 1}, DataType::f32,
            Device::CUDA);
  Tensor staging =
      zeros({static_cast<int64_t>(cfg.batch_size),
             static_cast<int64_t>(cfg.num_features)},
            DataType::f32, Device::CUDA);

  launch_mean_2d(mean.data(), input.data(), cfg.batch_size, cfg.num_features);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  DeviceBuffer l2_flush_buffer(l2_flush_bytes);

  for (int i = 0; i < warmup; ++i) {
    launch_variance(variance.data(), staging.data(), input.data(), mean.data(),
                    cfg.batch_size, cfg.num_features);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> latencies_ms;
  latencies_ms.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cudaMemsetAsync(l2_flush_buffer.data(), i,
                               l2_flush_buffer.size_bytes()));
    CHECK_CUDA(cudaEventRecord(start));
    launch_variance(variance.data(), staging.data(), input.data(), mean.data(),
                    cfg.batch_size, cfg.num_features);
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
  const double total_elems = static_cast<double>(cfg.batch_size) *
                             static_cast<double>(cfg.num_features);
  const double bytes_per_iter =
      bytes_per_iteration(cfg.batch_size, cfg.num_features);
  const double effective_gb_per_sec =
      (bytes_per_iter / (avg_ms / 1000.0)) / 1.0e9;

  const auto variance_host = variance.cpu();

  return {
      avg_ms,
      static_cast<double>(*min_it),
      static_cast<double>(*max_it),
      total_elems,
      effective_gb_per_sec,
      bytes_per_iter,
      static_cast<float *>(variance_host.data())[0],
  };
}

void print_case(const BenchmarkCase &cfg, const BenchmarkResult &result) {
  fmt::print(
      "{}={}  {}={}  {}={}  {}={}  {}={}  {}={}  {}={}  {}={}\n",
      paint("batch", ansi::kDim),
      paintf(ansi::kBoldBlue, "{:>6}", cfg.batch_size),
      paint("features", ansi::kDim),
      paintf(ansi::kBoldBlue, "{:>6}", cfg.num_features),
      paint("elements", ansi::kDim),
      paintf(ansi::kWhite, "{:>12.0f}", result.total_elems),
      paint("min_ms", ansi::kDim),
      paintf(ansi::kBoldGreen, "{:>9.6f}", result.min_ms),
      paint("avg_ms", ansi::kDim),
      paintf(ansi::kBoldYellow, "{:>9.6f}", result.avg_ms),
      paint("max_ms", ansi::kDim),
      paintf(ansi::kBoldRed, "{:>9.6f}", result.max_ms),
      paint("bandwidth_GBps", ansi::kDim),
      paintf(ansi::kBoldMagenta, "{:>9.3f}", result.effective_gb_per_sec),
      paint("variance", ansi::kDim),
      paintf(ansi::kBoldCyan, "{:.6f}", result.sample_variance));
}

} // namespace
} // namespace smollnet

int main(int argc, char **argv) {
  using namespace smollnet;

  const auto cfg = parse_args(argc, argv);
  const int warmup_iterations = std::max(cfg.warmup, kMinimumWarmupIterations);
  const size_t l2_flush_bytes = recommended_l2_flush_bytes();

  fmt::print("{}\n", paint("Variance kernel benchmark", ansi::kBoldCyan));
  fmt::print("{}={}  {}={}  {}={}\n", paint("iterations", ansi::kDim),
             paintf(ansi::kBoldYellow, "{}", cfg.iterations),
             paint("warmup", ansi::kDim),
             paintf(ansi::kBoldYellow, "{}", warmup_iterations),
             paint("l2_flush_bytes", ansi::kDim),
             paintf(ansi::kBoldMagenta, "{}", l2_flush_bytes));

  if (cfg.use_default_suite) {
    fmt::print("{}={} {}\n", paint("running_default_suite", ansi::kDim),
               paintf(ansi::kBoldGreen, "{}", kDefaultCases.size()),
               paint("cases", ansi::kWhite));
    for (const auto &bench_case : kDefaultCases) {
      const auto result =
          run_case(bench_case, cfg.iterations, warmup_iterations, l2_flush_bytes);
      print_case(bench_case, result);
    }
  } else {
    const auto result = run_case(cfg.single_case, cfg.iterations,
                                 warmup_iterations, l2_flush_bytes);
    print_case(cfg.single_case, result);
  }

  return 0;
}
