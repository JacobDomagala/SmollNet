#pragma once

#include "module.hpp"
#include "tensor.hpp"

namespace smollnet {

class LayerNorm : public Module{
  Tensor weights;
  Tensor bias;

public:
  LayerNorm();
  Tensor compute(const Tensor& t);
  Tensor operator()(const Tensor& t);

  Tensor forward(Tensor &t) override;
  void gradient_update() const override;
  void print() const override;
  std::vector<Tensor> parameters() const override;
};

} // namespace smollnet
