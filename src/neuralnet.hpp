#pragma once
#include "kernels.cuh"
#include "sgd.hpp"
#include "tensor.hpp"

#include <fmt/core.h>
#include <memory>
#include <vector>

namespace smollnet {

struct Module {
  virtual ~Module() = default;
  virtual Tensor forward(Tensor &t) const = 0;
  virtual void gradient_update() const = 0;
  virtual void print() const = 0;
};

struct Linear : Module {
  Linear(int64_t in_dim, int64_t out_dim) {
    weights = rand({in_dim, out_dim}, DataType::f32, Device::CUDA, true);
    bias = zeros({1, out_dim}, DataType::f32, Device::CUDA, true);
  }
  Tensor forward(Tensor &t) const override {
    return matmul(t, weights).add(bias);
  }
  void print() const override {
    printf("Linear layer [\n\tWeights: ");
    weights.print();

    printf("\tBias:");
    bias.print();
    printf("]\n");
  }

  void gradient_update() const override {
    launch_sgd_update(weights.data(), weights.grad().data(), 0.02,
                      weights.numel());
    weights.zero_grad();

    launch_sgd_update(bias.data(), bias.grad().data(), 0.02, bias.numel());
    bias.zero_grad();
  }

  Tensor weights;
  Tensor bias;
};

struct ReLU : Module {
  Tensor forward(Tensor &t) const override { return relu(t); }
  void gradient_update() const override {}
  void print() const override { printf("ReLU\n"); }
};

class Dense {
public:
  template <typename... Args> Dense(Args &&...modules) {
    (layers_.emplace_back(
         std::make_unique<std::decay_t<Args>>(std::forward<Args>(modules))),
     ...);
  }

  Tensor forward(const Tensor &input) const {
    Tensor output = input;
    for (auto &layer : layers_) {
      output = layer->forward(output);
    }
    return output;
  }

  void train(Tensor &input, Tensor &targets) const {
    constexpr int num_epochs = 32;

    auto features = input.copy();
    // std::vector<Tensor> tensors;
    // for(auto& l : layers_){
    //   auto&& params = l->params();
    //   tensors.insert(tensors.end(), params.begin(), params.end());
    // }
    // auto optim = SGD(std::move(tensors), 0.02);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
      auto output = forward(features);
      auto loss = mse(output, targets);
      loss.backward();
      for (auto &l : layers_) {
        l->gradient_update();
      }
      // optim.step();
      // optim.zero_grad();
      fmt::print("[{}] loss = {}\n", epoch,
                 static_cast<float *>(loss.cpu().data())[0]);
    }
  }

  void print() const noexcept {
    printf("Dense neural network [num_layers: %ld]\n", layers_.size());
    for (auto &layer : layers_) {
      layer->print();
    }
  }

private:
  std::vector<std::unique_ptr<Module>> layers_;
  Device device_ = Device::CUDA;
  DataType dtype_ = DataType::f32;
};

} // namespace smollnet
