#include "neuralnet.hpp"
#include "kernels.cuh"
#include "sgd.hpp"

#include <fmt/core.h>

namespace smollnet {

Linear::Linear(int64_t in_dim, int64_t out_dim) {
  weights = rand({in_dim, out_dim}, DataType::f32, Device::CUDA, true);
  bias = zeros({1, out_dim}, DataType::f32, Device::CUDA, true);
}

Tensor Linear::forward(Tensor &t) {
  return matmul(t, weights).add(bias);
}

void Linear::print() const {
  printf("Linear layer [\n\tWeights: ");
  weights.print();

  printf("\tBias:");
  bias.print();
  printf("]\n");
}

void Linear::gradient_update() const {
  if (weights.grad().initialized())
    launch_sgd_update(weights.data(), weights.grad().data(), 0.001f,
                      weights.numel());
  if (bias.grad().initialized())
    launch_sgd_update(bias.data(), bias.grad().data(), 0.001f, bias.numel());

  weights.zero_grad();
  bias.zero_grad();
}

std::vector<Tensor> Linear::parameters() const { return {weights, bias}; }

Tensor ReLU::forward(Tensor &t) { return relu(t); }
void ReLU::gradient_update() const {}
std::vector<Tensor> ReLU::parameters() const { return {}; }
void ReLU::print() const { printf("ReLU\n"); }

Tensor Dense::forward(const Tensor &input) const {
  Tensor output = input;
  for (auto &layer : layers_) {
    output = layer->forward(output);
  }
  return output;
}

std::vector<Tensor> Dense::parameters() const {
  std::vector<Tensor> params;

  for (auto &layer : layers_) {
    auto layer_params = layer->parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }

  return params;
}

void Dense::train(Tensor &input, Tensor &targets, Optimizer optimizer, float lr,
                  int32_t num_epochs) const {
  auto features = input.copy();

  auto optim = SGD(std::move(parameters()), lr);

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto output = forward(features);
    auto loss = mse(output, targets);
    loss.backward();

    optim.step();
    optim.zero_grad();
    fmt::print("[{}] loss = {}\n", epoch,
               static_cast<float *>(loss.cpu().data())[0]);
  }
}

void Dense::print() const noexcept {
  printf("Dense neural network [num_layers: %ld]\n", layers_.size());
  for (auto &layer : layers_) {
    layer->print();
  }
}

} // namespace smollnet
