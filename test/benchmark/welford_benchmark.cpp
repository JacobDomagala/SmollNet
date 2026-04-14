#include "benchmark_harness.hpp"
#include "helpers.hpp"
#include "kernels.cuh"
#include "tensor.hpp"
#include "welford_internal.inl"

#include <array>
#include <cstdint>
#include <cstdlib>

#include <fmt/core.h>

namespace smollnet {
namespace {

struct WelfordBenchmarkMode {
  const char *label;
  int32_t dim;
};

struct WelfordStageEntry {
  int count;
  float mean;
  float m2;
};

static_assert(sizeof(WelfordStageEntry) ==
              sizeof(int) + 2 * sizeof(float));

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

constexpr std::array<WelfordBenchmarkMode, 2> kModes = {{
    {"row", 0},
    {"column", 1},
}};

size_t ceil_div(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

size_t output_elements(const BenchmarkCase &cfg,
                       const WelfordBenchmarkMode &mode) {
  return mode.dim == 0 ? cfg.batch_size : cfg.num_features;
}

BenchmarkConfig parse_args(int argc, char **argv) {
  BenchmarkConfig cfg;

  if (argc == 1) {
    return cfg;
  }

  ASSERT(argc == 3 || argc == 5,
         "Usage: welford_benchmark "
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

double bytes_per_iteration(const BenchmarkCase &cfg,
                           const WelfordBenchmarkMode &mode) {
  const double input_bytes = static_cast<double>(cfg.batch_size) *
                             static_cast<double>(cfg.num_features) *
                             sizeof(float);
  const double stage_entries =
      mode.dim == 0
          ? static_cast<double>(ceil_div(cfg.num_features,
                                         welford_internal::kBlockDim)) *
                static_cast<double>(cfg.batch_size)
          : static_cast<double>(cfg.num_features) *
                static_cast<double>(
                    ceil_div(cfg.batch_size, welford_internal::kColChunkSize));
  const double staging_bytes =
      stage_entries * static_cast<double>(sizeof(WelfordStageEntry));
  const double output_bytes =
      static_cast<double>(output_elements(cfg, mode)) * sizeof(float);

  // launch_welford is two-pass for both dim=0 and dim=1:
  // 1) read input, write Welford staging tuples
  // 2) read staging tuples, write final variance
  return input_bytes + staging_bytes + staging_bytes + output_bytes;
}

struct BenchmarkResult {
  double avg_ms;
  double min_ms;
  double max_ms;
  double total_elems;
  double effective_gb_per_sec;
  double bytes_per_iter;
};

BenchmarkResult run_case(const BenchmarkCase &cfg,
                         const WelfordBenchmarkMode &mode,
                         const bench::RunConfig &run_cfg) {
  Tensor input = rand({static_cast<int64_t>(cfg.batch_size),
                       static_cast<int64_t>(cfg.num_features)},
                      DataType::f32, Device::CUDA);

  const int64_t output_dims[2] = {
      mode.dim == 0 ? static_cast<int64_t>(cfg.batch_size) : 1,
      mode.dim == 0 ? 1 : static_cast<int64_t>(cfg.num_features),
  };
  Tensor variance = zeros(output_dims, DataType::f32, Device::CUDA);

  const auto timing = bench::measure_cuda_operation(run_cfg, [&] {
    launch_welford(input.data(), variance.data(), cfg.num_features,
                   cfg.batch_size, mode.dim,
                   WelfordType::PopulationVariance);
  });

  const double total_elems = static_cast<double>(cfg.batch_size) *
                             static_cast<double>(cfg.num_features);
  const double bytes_per_iter = bytes_per_iteration(cfg, mode);
  const double effective_gb_per_sec =
      (bytes_per_iter / (timing.avg_ms / 1000.0)) / 1.0e9;

  return {
      timing.avg_ms,
      timing.min_ms,
      timing.max_ms,
      total_elems,
      effective_gb_per_sec,
      bytes_per_iter,
  };
}

void print_case(const WelfordBenchmarkMode &mode, const BenchmarkCase &cfg,
                const BenchmarkResult &result) {
  bench::print_fields({
      bench::field("axis", bench::ansi::kBoldCyan, "{}", mode.label),
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
  });
}

} // namespace
} // namespace smollnet

int main(int argc, char **argv) {
  using namespace smollnet;

  auto cfg = parse_args(argc, argv);
  cfg.run.l2_flush_bytes = bench::recommended_l2_flush_bytes();

  bench::print_banner("Welford variance benchmark", cfg.run);

  if (cfg.use_default_suite) {
    bench::print_fields({
        bench::field("suite_cases", bench::ansi::kBoldGreen, "{}",
                     kDefaultCases.size()),
        bench::field("axes", bench::ansi::kBoldCyan, "row,column"),
    });
    for (const auto &bench_case : kDefaultCases) {
      for (const auto &mode : kModes) {
        const auto result = run_case(bench_case, mode, cfg.run);
        print_case(mode, bench_case, result);
      }
    }
  } else {
    for (const auto &mode : kModes) {
      const auto result = run_case(cfg.single_case, mode, cfg.run);
      print_case(mode, cfg.single_case, result);
    }
  }

  return 0;
}
