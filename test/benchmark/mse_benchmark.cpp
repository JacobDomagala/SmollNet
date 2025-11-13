#include "benchmark_harness.hpp"
#include "helpers.hpp"
#include "kernels.cuh"
#include "tensor.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <fmt/core.h>

namespace smollnet {
namespace {

constexpr int kMseBlockSize = 256;

struct BenchmarkCase {
  size_t elements;
};

struct BenchmarkConfig {
  BenchmarkCase single_case{1ull << 24};
  bench::RunConfig run;
  bool use_default_suite = true;
};

constexpr std::array<BenchmarkCase, 7> kDefaultCases = {{
    {1ull << 12},
    {1ull << 16},
    {1ull << 18},
    {1ull << 20},
    {1ull << 22},
    {1ull << 24},
    {1ull << 25},
}};

size_t parse_size_arg(const char *text, const char *name) {
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);

  ASSERT(end != nullptr && *end == '\0',
         fmt::format("Invalid {} value '{}'", name, text));
  ASSERT(value > 0, fmt::format("{} must be greater than zero", name));
  ASSERT(value <= static_cast<unsigned long long>(
                      std::numeric_limits<int64_t>::max()),
         fmt::format("{} exceeds the maximum supported tensor size", name));

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

BenchmarkConfig parse_args(int argc, char **argv) {
  BenchmarkConfig cfg;

  if (argc == 1) {
    return cfg;
  }

  ASSERT(argc == 2 || argc == 4,
         "Usage: mse_benchmark [elements [iterations warmup]]");

  cfg.use_default_suite = false;
  cfg.single_case.elements = parse_size_arg(argv[1], "elements");

  if (argc == 4) {
    cfg.run.iterations = parse_int_arg(argv[2], "iterations");
    cfg.run.warmup = parse_int_arg(argv[3], "warmup");
  }

  return cfg;
}

size_t ceil_div(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

size_t blocks_per_launch(const BenchmarkCase &cfg) {
  return ceil_div(cfg.elements, static_cast<size_t>(kMseBlockSize));
}

double input_bytes_per_iteration(const BenchmarkCase &cfg) {
  return static_cast<double>(cfg.elements) * 2.0 * sizeof(float);
}

double flops_per_iteration(const BenchmarkCase &cfg) {
  const double element_flops = static_cast<double>(cfg.elements) * 3.0;
  const double reduction_flops =
      static_cast<double>(blocks_per_launch(cfg)) * kMseBlockSize;

  return element_flops + reduction_flops + 1.0;
}

std::string format_scaled(double value, const char *base_unit) {
  constexpr std::array<const char *, 6> prefixes = {{
      "",
      "K",
      "M",
      "G",
      "T",
      "P",
  }};

  size_t unit_idx = 0;
  while (value >= 1000.0 && unit_idx + 1 < prefixes.size()) {
    value /= 1000.0;
    ++unit_idx;
  }

  return fmt::format("{:.3f} {}{}", value, prefixes[unit_idx], base_unit);
}

struct BenchmarkResult {
  double avg_ms;
  double min_ms;
  double max_ms;
  double input_bytes_per_iter;
  double flops_per_iter;
  double effective_input_gb_per_sec;
  double flops_per_sec;
  double ns_per_elem;
  size_t blocks;
};

BenchmarkResult run_case(const BenchmarkCase &cfg,
                         const bench::RunConfig &run_cfg) {
  Tensor pred = rand({static_cast<int64_t>(cfg.elements)}, DataType::f32,
                     Device::CUDA);
  Tensor target = rand({static_cast<int64_t>(cfg.elements)}, DataType::f32,
                       Device::CUDA);
  Tensor loss = zeros({1}, DataType::f32, Device::CUDA);

  const auto timing = bench::measure_cuda_operation(run_cfg, [&] {
    launch_mse(loss.data(), loss.dtype(), pred.data(), target.data(),
               pred.dtype(), cfg.elements);
  });

  const double total_elems = static_cast<double>(cfg.elements);
  const double input_bytes_per_iter = input_bytes_per_iteration(cfg);
  const double flops_per_iter = flops_per_iteration(cfg);
  const double effective_input_gb_per_sec =
      (input_bytes_per_iter / (timing.avg_ms / 1000.0)) / 1.0e9;
  const double flops_per_sec = flops_per_iter / (timing.avg_ms / 1000.0);
  const double ns_per_elem = (timing.avg_ms * 1.0e6) / total_elems;

  return {
      timing.avg_ms,
      timing.min_ms,
      timing.max_ms,
      input_bytes_per_iter,
      flops_per_iter,
      effective_input_gb_per_sec,
      flops_per_sec,
      ns_per_elem,
      blocks_per_launch(cfg),
  };
}

void print_case(const BenchmarkCase &cfg, const BenchmarkResult &result) {
  bench::print_fields({
      bench::field("elements", bench::ansi::kBoldBlue, "{:>12}",
                   cfg.elements),
      bench::field("blocks", bench::ansi::kBoldCyan, "{:>9}",
                   result.blocks),
      bench::field("input_bytes", bench::ansi::kWhite, "{:>12.0f}",
                   result.input_bytes_per_iter),
      bench::field("flops", bench::ansi::kWhite, "{:>13}",
                   format_scaled(result.flops_per_iter, "FLOP")),
      bench::field("min_ms", bench::ansi::kBoldGreen, "{:>9.6f}",
                   result.min_ms),
      bench::field("avg_ms", bench::ansi::kBoldYellow, "{:>9.6f}",
                   result.avg_ms),
      bench::field("max_ms", bench::ansi::kBoldRed, "{:>9.6f}",
                   result.max_ms),
      bench::field("ns_per_elem", bench::ansi::kBoldMagenta, "{:>9.4f}",
                   result.ns_per_elem),
      bench::field("FLOP/s", bench::ansi::kBoldMagenta, "{:>13}",
                   format_scaled(result.flops_per_sec, "FLOP/s")),
      bench::field("input_GBps", bench::ansi::kBoldMagenta, "{:>9.3f}",
                   result.effective_input_gb_per_sec),
  });
}

} // namespace
} // namespace smollnet

int main(int argc, char **argv) {
  using namespace smollnet;

  auto cfg = parse_args(argc, argv);
  cfg.run.l2_flush_bytes = bench::recommended_l2_flush_bytes();

  bench::print_banner("MSE benchmark", cfg.run);

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
