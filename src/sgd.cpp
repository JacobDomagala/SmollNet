#include "sgd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <fmt/format.h>

namespace smollnet {

void SGD::step() const {
  for (const auto &p : params_) {
    ASSERT(
        p.size(0) == p.grad().size(0),
        fmt::format("Size 0 mismatch!: {} vs {}", p.size(0), p.grad().size(0)));
    ASSERT(
        p.size(1) == p.grad().size(1),
        fmt::format("Size 1 mismatch!: {} vs {}", p.size(1), p.grad().size(1)));
    ASSERT(
        p.size(2) == p.grad().size(2),
        fmt::format("Size 2 mismatch!: {} vs {}", p.size(2), p.grad().size(2)));

    launch_sgd_update(p.data(), p.grad().data(), lr_, p.numel());
  }
}

void SGD::zero_grad() const {
  for (const auto &p : params_) {
    p.zero_grad();
  }
}

} // namespace smollnet
