#pragma once

#include "tensor.hpp"

namespace smollnet {

class LayerNorm {

public:
  LayerNorm();
  Tensor compute(const Tensor& t);
  Tensor operator()(const Tensor& t);
};

} // namespace smollnet
