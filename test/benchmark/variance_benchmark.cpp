#include "helpers.hpp"
#include "kernels.cuh"
#include "tensor.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>

#include <fmt/core.h>

namespace smollnet {
namespace {

struct BenchmarkCase {
  size_t batch_size;
  size_t num_features;
};

struct BenchmarkConfig {
  BenchmarkCase single_case{4096, 1024};
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
    {32, 128},
    {256, 128},
    {1024, 128},
    {32, 1024},
    {256, 1024},
    {1024, 1024},
    {256, 4096},
    {1024, 4096},
}};

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

double bytes_per_iteration(size_t batch_size, size_t num_features) {
  const size_t total = batch_size * num_features;

  // Step 1: read input + mean, write staging.
  // Step 2: read staging, write variance.
  return static_cast<double>((4 * total + batch_size) * sizeof(float));
}

struct BenchmarkResult {
  double avg_ms;
  double total_elems;
  double gib_per_sec;
  float sample_variance;
};

BenchmarkResult run_case(const BenchmarkCase &cfg, int iterations, int warmup) {
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

  for (int i = 0; i < warmup; ++i) {
    launch_variance(variance.data(), staging.data(), input.data(), mean.data(),
                    cfg.batch_size, cfg.num_features);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_variance(variance.data(), staging.data(), input.data(), mean.data(),
                    cfg.batch_size, cfg.num_features);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  const double avg_ms = total_ms / static_cast<double>(iterations);
  const double total_elems =
      static_cast<double>(cfg.batch_size) * static_cast<double>(cfg.num_features);
  const double gib_per_sec =
      (bytes_per_iteration(cfg.batch_size, cfg.num_features) /
       (avg_ms / 1000.0)) /
      (1024.0 * 1024.0 * 1024.0);

  const auto variance_host = variance.cpu();

  return {
      avg_ms,
      total_elems,
      gib_per_sec,
      static_cast<float *>(variance_host.data())[0],
  };
}

void print_case(const BenchmarkCase &cfg, const BenchmarkResult &result) {
  fmt::print(
      "batch_size={:>5} num_features={:>5} elements={:>10.0f} "
      "avg_time_ms={:>9.6f} approx_gib_per_sec={:>8.3f} sample_variance={:.6f}\n",
      cfg.batch_size, cfg.num_features, result.total_elems, result.avg_ms,
      result.gib_per_sec, result.sample_variance);
}

} // namespace
} // namespace smollnet

int main(int argc, char **argv) {
  using namespace smollnet;

  const auto cfg = parse_args(argc, argv);

  fmt::print("Variance kernel benchmark\n");
  fmt::print("iterations={} warmup={}\n", cfg.iterations, cfg.warmup);

  if (cfg.use_default_suite) {
    fmt::print("running_default_suite={} cases\n", kDefaultCases.size());
    for (const auto &bench_case : kDefaultCases) {
      const auto result = run_case(bench_case, cfg.iterations, cfg.warmup);
      print_case(bench_case, result);
    }
  } else {
    const auto result = run_case(cfg.single_case, cfg.iterations, cfg.warmup);
    print_case(cfg.single_case, result);
  }

  return 0;
}
