#pragma once

#include <vector>

namespace smollnet {

class Tensor;

struct Module {
  virtual ~Module() = default;
  virtual Tensor forward(Tensor &t) = 0;
  virtual void print() const = 0;
  virtual std::vector<Tensor> parameters() const = 0;
};

} // namespace smollnet
