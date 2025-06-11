#include "layer_norm.hpp"
#include "kernels.cuh"
#include "autograd.hpp"

#include <fmt/format.h>

namespace smollnet {

LayerNorm::LayerNorm() {}

Tensor LayerNorm::operator()(const Tensor &t) { return compute(t); }

Tensor LayerNorm::compute(const Tensor &t) {
  if (!weights.initialized()) {
    weights = ones({t.size(1), 1}, t.dtype(), t.device());
  }

  if (!bias.initialized()) {
    bias = zeros({t.size(1), 1}, t.dtype(), t.device());
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
    meta->grad_fn = std::make_shared<LayerNormFunction>(mean, variance, normalized, weights);
  }

  return normalized;
}

Tensor LayerNorm::forward(Tensor &t) { return compute(t); }

void LayerNorm::gradient_update() const {
  //   if (weights.grad().initialized())
  //     launch_sgd_update(weights.data(), weights.grad().data(), 1e-5f,
  //                       weights.numel());
  //   if (bias.grad().initialized())
  //     launch_sgd_update(bias.data(), bias.grad().data(), 1e-5f,
  //     bias.numel());

  //   weights.zero_grad();
  //   bias.zero_grad();
}

void LayerNorm::print() const { fmt::print("LayerNorm"); }

std::vector<Tensor> LayerNorm::parameters() const { return {weights, bias}; }

} // namespace smollnet
