#include "sgd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <fmt/format.h>

namespace smollnet {

void SGD::step() const {
  for (const auto &p : params_) {
    Tensor grad = p.grad();
    ASSERT(p.ndims() == grad.ndims(),
           fmt::format("Rank mismatch!: {} vs {}", p.ndims(), grad.ndims()));
    for (int64_t dim = 0; dim < p.ndims(); ++dim) {
      ASSERT(p.size(dim) == grad.size(dim),
             fmt::format("Size {} mismatch!: {} vs {}", dim, p.size(dim),
                         grad.size(dim)));
    }

    launch_sgd_update(p.data(), grad.data(), p.dtype(), grad.dtype(), lr_,
                      p.numel());
  }
}

void SGD::zero_grad() const {
  for (const auto &p : params_) {
    p.zero_grad();
  }
}

} // namespace smollnet
