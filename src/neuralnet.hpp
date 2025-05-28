#pragma once

#include "tensor.hpp"

#include <memory>
#include <vector>

namespace smollnet {

struct Module {
  virtual ~Module() = default;
  virtual Tensor forward(Tensor &t) const = 0;
  virtual void gradient_update() const = 0;
  virtual void print() const = 0;
  virtual std::vector<Tensor> parameters() const = 0;
};

struct Linear : Module {
  Linear(int64_t in_dim, int64_t out_dim);

  Tensor forward(Tensor &t) const override;
  std::vector<Tensor> parameters() const override;
  void print() const override;

  void gradient_update() const override;

  Tensor weights;
  Tensor bias;
};

struct ReLU : Module {
  Tensor forward(Tensor &t) const override;
  void gradient_update() const override;
  void print() const override;
  std::vector<Tensor> parameters() const override;
};

class Dense {
public:
  template <typename... Args> Dense(Args &&...modules) {
    (layers_.emplace_back(
         std::make_unique<std::decay_t<Args>>(std::forward<Args>(modules))),
     ...);
  }

  Tensor forward(const Tensor &input) const;
  std::vector<Tensor> parameters() const;

  void train(Tensor &input, Tensor &targets,
             Optimizer optimizer = Optimizer::SGD, float lr = 0.0001f,
             int32_t num_epochs = 32) const;

  void print() const noexcept;

private:
  std::vector<std::unique_ptr<Module>> layers_;
  Device device_ = Device::CUDA;
  DataType dtype_ = DataType::f32;
};

} // namespace smollnet
