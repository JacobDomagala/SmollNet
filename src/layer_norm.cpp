#include "layer_norm.hpp"
#include "autograd.hpp"
#include "dtype_utils.hpp"
#include "kernels.cuh"

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

  const DataType stats_dtype = accumulation_dtype(t.dtype());

  auto mean = zeros({t.size(0), 1}, stats_dtype, t.device());
  launch_mean_2d(mean.data(), mean.dtype(), t.data(), t.dtype(), t.size(0),
                 t.size(1));

  auto variance = zeros({t.size(0), 1}, stats_dtype, t.device());
  launch_welford(t.data(), t.dtype(), variance.data(), variance.dtype(),
                 t.size(1), t.size(0), 0, WelfordType::PopulationVariance);

  auto normalized = zeros(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                          t.requires_grad());
  launch_layer_norm(normalized.data(), t.data(), mean.data(), variance.data(),
                    weights.data(), bias.data(), normalized.dtype(),
                    mean.dtype(), t.size(0), t.size(1));

  if (normalized.requires_grad()) {
    auto *meta = normalized.autograd();

    meta->is_leaf = false;
    meta->grad_fn = std::make_shared<LayerNormFunction>(
        mean, variance, normalized, t, weights, bias);
  }

  return normalized;
}

Tensor LayerNorm::forward(Tensor &t) { return compute(t); }

void LayerNorm::print() const { fmt::print("LayerNorm"); }

std::vector<Tensor> LayerNorm::parameters() const { return {weights, bias}; }

} // namespace smollnet
