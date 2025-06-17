#include "neuralnet.hpp"
#include "sgd.hpp"

#include <fmt/core.h>

namespace smollnet {

Linear::Linear(int64_t in_dim, int64_t out_dim)
    : weights(rand({in_dim, out_dim}, DataType::f32, Device::CUDA, true)),
      bias(zeros({1, out_dim}, DataType::f32, Device::CUDA, true)) {}

Tensor Linear::forward(Tensor &t) { return matmul(t, weights).add(bias); }

void Linear::print() const {
  printf("Linear layer [\n\tWeights: ");
  weights.print();

  printf("\tBias:");
  bias.print();
  printf("]\n");
}

std::vector<Tensor> Linear::parameters() const { return {weights, bias}; }

Tensor ReLU::forward(Tensor &t) { return relu(t); }
std::vector<Tensor> ReLU::parameters() const { return {}; }
void ReLU::print() const { printf("ReLU\n"); }

Tensor GeLU::forward(Tensor &t) { return gelu(t); }
std::vector<Tensor> GeLU::parameters() const { return {}; }
void GeLU::print() const { printf("GeLU\n"); }

Tensor Dense::forward(const Tensor &input) const {
  Tensor output = input;
  for (const auto &layer : layers_) {
    output = layer->forward(output);
  }
  return output;
}

std::vector<Tensor> Dense::parameters() const {
  std::vector<Tensor> params;

  for (const auto &layer : layers_) {
    auto layer_params = layer->parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }

  return params;
}

void Dense::train(const Tensor &input, const Tensor &targets, const float lr,
                  const int32_t num_epochs) const {
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
  fmt::print("Dense neural network [num_layers: {}]\n", layers_.size());
  for (const auto &layer : layers_) {
    layer->print();
  }
}

} // namespace smollnet
