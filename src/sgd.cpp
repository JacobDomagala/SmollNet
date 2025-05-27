#include "sgd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

namespace smollnet {

void SGD::step() {
  for (auto &p : params_) {
    launch_sgd_update(p.data(), p.grad().data(), lr_, p.numel());
  }
}

void SGD::zero_grad() {
  for (auto &p : params_) {
    p.zero_grad();
  }
}

} // namespace smollnet
