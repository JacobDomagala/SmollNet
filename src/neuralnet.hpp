#pragma once

#include "module.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace smollnet {

struct Linear : Module {
  Linear(int64_t in_dim, int64_t out_dim);

  Tensor forward(Tensor &t) override;
  std::vector<Tensor> parameters() const override;
  void print() const override;

  Tensor weights;
  Tensor bias;
};

struct ReLU : Module {
  Tensor forward(Tensor &t) override;
  void print() const override;
  std::vector<Tensor> parameters() const override;
};

struct GeLU : Module {
  Tensor forward(Tensor &t) override;
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

  explicit Dense(std::vector<std::unique_ptr<Module>>&& modules);

  Tensor forward(const Tensor &input) const;
  std::vector<Tensor> parameters() const;

  void train(const Tensor &input, const Tensor &targets,
             const float lr = 0.0001f, const int32_t num_epochs = 32) const;

  void print() const;
  void print_grads() const;

private:
  std::vector<std::unique_ptr<Module>> layers_;
  Device device_ = Device::CUDA;
  DataType dtype_ = DataType::f32;
};

} // namespace smollnet
