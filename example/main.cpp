#include <helpers.hpp>
#include <tensor.hpp>
#include <neuralnet.hpp>
#include <types.hpp>

#include <cuda_runtime.h>

using namespace smollnet;

int main() {
  // contents: uninitialized float32 on GPU
  Tensor a = rand({4, 8}, DataType::f32, Device::CUDA);
  a.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  // contents: all ones, shape [2, 3], stride [3, 1]
  Tensor b = ones({4, 8}, DataType::f32, Device::CUDA);
  b.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  // elementwise addition, c[i,j] = a[i,j] + b[i,j]
  Tensor c = a + b;
  c.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  // view: shape [3, 2], stride adjusted, no data copied
  Tensor d = c.transpose(0, 1);
  d.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  Tensor e = c - b;
  e.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  Tensor f = sum(e, 0);
  f.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  Tensor g = sum(f, 0);
  g.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  Tensor h = matmul(b, d);
  h.print();
  CHECK_CUDA(cudaDeviceSynchronize());

  Tensor h_host = h.cpu();
  CHECK_CUDA(cudaDeviceSynchronize());

  auto net = Dense(Linear(32, 12), ReLU(), Linear(12,1));
  net.print();
}
