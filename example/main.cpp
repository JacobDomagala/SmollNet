#include <helpers.hpp>
#include <tensor.hpp>
#include <neuralnet.hpp>
#include <types.hpp>

#include <cuda_runtime.h>

using namespace smollnet;

int main() {
  // contents: uninitialized float32 on GPU
  Tensor a = rand({1, 32}, DataType::f32, Device::CUDA);
  a.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  auto net = Dense(Linear(32, 12), ReLU(), Linear(12,1));
  auto res = net.forward(a);
  net.print();
  res.print();
}
