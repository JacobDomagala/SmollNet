#pragma once

#include "module.hpp"
#include "tensor.hpp"

namespace smollnet {

struct LayerNorm : public Module {
  Tensor compute(const Tensor &t);
  Tensor operator()(const Tensor &t);

  Tensor forward(Tensor &t) override;
  void print() const override;
  std::vector<Tensor> parameters() const override;

  Tensor weights;
  Tensor bias;
};

} // namespace smollnet
