#include <helpers.hpp>
#include <tensor.hpp>
#include <neuralnet.hpp>
#include <types.hpp>

#include <cuda_runtime.h>
#include <fmt/core.h>

using namespace smollnet;

int main() {
  Tensor input = rand({1, 128}, DataType::f32, Device::CUDA);
  Tensor targets = rand({1, 1}, DataType::f32, Device::CUDA);

  auto net = Dense(Linear(128, 64), ReLU(), Linear(64,1));
  net.train(input, targets);
}
