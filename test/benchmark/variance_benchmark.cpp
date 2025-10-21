#include "benchmark_harness.hpp"
#include "helpers.hpp"
#include "kernels.cuh"
#include "tensor.hpp"

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
  BenchmarkCase single_case{8192, 4096};
  bench::RunConfig run;
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
    cfg.run.iterations = parse_int_arg(argv[3], "iterations");
    cfg.run.warmup = parse_int_arg(argv[4], "warmup");
  }

  return cfg;
}

double bytes_per_iteration(size_t batch_size, size_t num_features) {
  const double total =
      static_cast<double>(batch_size) * static_cast<double>(num_features);

  // Counts only launch_variance traffic, not the one-time mean precompute.
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

BenchmarkResult run_case(const BenchmarkCase &cfg,
                         const bench::RunConfig &run_cfg) {
  Tensor input = rand({static_cast<int64_t>(cfg.batch_size),
                       static_cast<int64_t>(cfg.num_features)},
                      DataType::f32, Device::CUDA);

  Tensor variance = zeros({static_cast<int64_t>(cfg.batch_size), 1},
                          DataType::f32, Device::CUDA);

  const auto timing = bench::measure_cuda_operation(run_cfg, [&] {
    launch_welford(input.data(), variance.data(), cfg.num_features,
                   cfg.batch_size, 0, WelfordType::PopulationVariance);
  });

  const double total_elems = static_cast<double>(cfg.batch_size) *
                             static_cast<double>(cfg.num_features);
  const double bytes_per_iter =
      bytes_per_iteration(cfg.batch_size, cfg.num_features);
  const double effective_gb_per_sec =
      (bytes_per_iter / (timing.avg_ms / 1000.0)) / 1.0e9;

  const auto variance_host = variance.cpu();

  return {
      timing.avg_ms,
      timing.min_ms,
      timing.max_ms,
      total_elems,
      effective_gb_per_sec,
      bytes_per_iter,
      static_cast<float *>(variance_host.data())[0],
  };
}

void print_case(const BenchmarkCase &cfg, const BenchmarkResult &result) {
  bench::print_fields({
      bench::field("batch", bench::ansi::kBoldBlue, "{:>6}", cfg.batch_size),
      bench::field("features", bench::ansi::kBoldBlue, "{:>6}",
                   cfg.num_features),
      bench::field("elements", bench::ansi::kWhite, "{:>12.0f}",
                   result.total_elems),
      bench::field("bytes_per_iter", bench::ansi::kWhite, "{:>12.0f}",
                   result.bytes_per_iter),
      bench::field("min_ms", bench::ansi::kBoldGreen, "{:>9.6f}",
                   result.min_ms),
      bench::field("avg_ms", bench::ansi::kBoldYellow, "{:>9.6f}",
                   result.avg_ms),
      bench::field("max_ms", bench::ansi::kBoldRed, "{:>9.6f}", result.max_ms),
      bench::field("bandwidth_GBps", bench::ansi::kBoldMagenta, "{:>9.3f}",
                   result.effective_gb_per_sec),
      bench::field("variance", bench::ansi::kBoldCyan, "{:.6f}",
                   result.sample_variance),
  });
}

} // namespace
} // namespace smollnet

int main(int argc, char **argv) {
  using namespace smollnet;

  auto cfg = parse_args(argc, argv);
  cfg.run.l2_flush_bytes = bench::recommended_l2_flush_bytes();

  bench::print_banner("Variance kernel benchmark", cfg.run);

  if (cfg.use_default_suite) {
    bench::print_fields({
        bench::field("suite_cases", bench::ansi::kBoldGreen, "{}",
                     kDefaultCases.size()),
    });
    for (const auto &bench_case : kDefaultCases) {
      const auto result = run_case(bench_case, cfg.run);
      print_case(bench_case, result);
    }
  } else {
    const auto result = run_case(cfg.single_case, cfg.run);
    print_case(cfg.single_case, result);
  }

  return 0;
}
