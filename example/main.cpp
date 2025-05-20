#include <helpers.hpp>
#include <tensor.hpp>
#include <neuralnet.hpp>
#include <types.hpp>

#include <cuda_runtime.h>
#include <fmt/core.h>

using namespace smollnet;

int main() {
  // contents: uninitialized float32 on GPU
  Tensor a = rand({1, 32}, DataType::f32, Device::CUDA);

  auto net = Dense(Linear(32, 12), ReLU(), Linear(12,1));
  auto res = net.forward(a).cpu();

  res.print();

  fmt::print("Final value: {}\n", *static_cast<float*>(res.data()));
}
