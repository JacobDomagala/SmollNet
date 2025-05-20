#pragma once
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace smollnet {

struct Module {
  virtual ~Module() = default;
  virtual Tensor forward(Tensor &t) = 0;
  virtual void print() const = 0;
};

struct Linear : Module {
  Linear(int64_t in_dim, int64_t out_dim) {
    weights = rand({in_dim, out_dim}, DataType::f32, Device::CUDA);
    bias = zeros({1, out_dim}, DataType::f32, Device::CUDA);
  }
  Tensor forward(Tensor &t) override { return matmul(t, weights).add(bias); }
  void print() const override {
    printf("Linear layer [\n\tWeights: ");
    weights.print();

    printf("\tBias:");
    bias.print();
    printf("]\n");
  }

  Tensor weights;
  Tensor bias;
};

struct ReLU : Module {
  Tensor forward(Tensor &t) override { return relu(t); }
  void print() const override {
    printf("ReLU\n");
  }
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
    for(auto& layer : layers_){
        output = layer->forward(output);
    }
    return output;
  }

  void print() const noexcept{
    printf("Dense neural network [num_layers: %ld]\n", layers_.size());
    for(auto& layer : layers_){
      layer->print();
    }
  }

private:
  std::vector<std::unique_ptr<Module>> layers_;
  Device device_ = Device::CPU;
  DataType dtype_ = DataType::f32;
};

} // namespace smollnet
