#pragma once

#include "tensor.hpp"
#include <vector>

namespace smollnet {

class SGD {
  std::vector<Tensor> params_;
  float lr_;

public:
  SGD(std::vector<Tensor> &&params, float lr)
      : params_(std::move(params)), lr_(lr) {}
  void step() const;
  void zero_grad() const;
};

} // namespace smollnet
