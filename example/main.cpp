#include <helpers.hpp>
#include <tensor.hpp>
#include <neuralnet.hpp>
#include <types.hpp>

#include <cuda_runtime.h>
#include <fmt/core.h>

using namespace smollnet;

int main() {
  // contents: uninitialized float32 on GPU
  Tensor a = rand({1, 128}, DataType::f32, Device::CUDA);

  auto net = Dense(Linear(128, 64), ReLU(), Linear(64,1));
  auto res = net.forward(a);
  res.backward();

  res.print();

  fmt::print("Final value: {}\n", *static_cast<float*>(res.cpu().data()));
}
