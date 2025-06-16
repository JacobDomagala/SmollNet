#include "layer_norm.hpp"
#include "kernels.cuh"
#include "autograd.hpp"

#include <fmt/format.h>

namespace smollnet {

Tensor LayerNorm::operator()(const Tensor &t) { return compute(t); }

Tensor LayerNorm::compute(const Tensor &t) {
  if (!weights.initialized()) {
    weights = ones({t.size(1), 1}, t.dtype(), t.device(), true);
  }

  if (!bias.initialized()) {
    bias = zeros({t.size(1), 1}, t.dtype(), t.device(), true);
  }

  auto mean = zeros({t.size(0), 1}, t.dtype(), t.device());
  launch_mean_2d(mean.data(), t.data(), t.size(0), t.size(1));

  auto variance = zeros({t.size(0), 1}, t.dtype(), t.device());
  auto staging = zeros(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                       t.requires_grad());
  launch_variance(variance.data(), staging.data(), t.data(), mean.data(),
                  t.size(0), t.size(1));

  auto normalized = zeros(t.dims().data(), t.ndims(), t.dtype(), t.device(), t.requires_grad());
  launch_layer_norm(normalized.data(), t.data(), mean.data(), variance.data(),
                    weights.data(), bias.data(), t.size(0), t.size(1));

  if(normalized.requires_grad()) {
    auto* meta = normalized.autograd();

    meta->is_leaf = false;
    meta->grad_fn = std::make_shared<LayerNormFunction>(mean, variance, normalized, t, weights, bias);
  }

  return normalized;
}

Tensor LayerNorm::forward(Tensor &t) { return compute(t); }

void LayerNorm::print() const { fmt::print("LayerNorm"); }

std::vector<Tensor> LayerNorm::parameters() const { return {weights, bias}; }

} // namespace smollnet
