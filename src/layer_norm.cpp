#include "layer_norm.hpp"
#include "kernels.cuh"

namespace smollnet {

LayerNorm::LayerNorm() {}
Tensor LayerNorm::operator()(const Tensor& t) {
    return compute(t);
}
Tensor LayerNorm::compute(const Tensor &t) {
    // Compute the mean:
    auto mean = sum(t, 1);

    // Compute variance
    // total_sum=0
    // For xi in features:
    //  total_rum += (mean - xi)^2
    //
    // total_sum /= features.size()

    return mean;
}

} // namespace smollnet
