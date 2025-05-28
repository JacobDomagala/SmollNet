#include <smollnet.hpp>

#include <cuda_runtime.h>
#include <fmt/core.h>

using namespace smollnet;

int main() {
  Tensor input = rand({1, 128}, DataType::f32, Device::CUDA);
  Tensor targets = rand({1, 1}, DataType::f32, Device::CUDA);

  auto net = Dense(Linear(128, 64), ReLU(), Linear(64, 1));

  // Single forward pass
  auto res = net.forward(input);
  auto loss = mse(res, targets);

  // Single nackward pass
  loss.backward();

  auto optim = SGD(net.parameters(), 0.0001f);
  optim.step();
  optim.zero_grad();

  // RE-evaluate error
  res = net.forward(input);
  auto new_loss = mse(res, targets);

  fmt::print("Loss before gradient update: {} and after {}\n",
             static_cast<float *>(loss.cpu().data())[loss.numel()],
             static_cast<float *>(new_loss.cpu().data())[new_loss.numel()]);
}
