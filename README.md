# SmollNet

SmollNet is a small deep learning library written in pure CUDA/C++


Example usage:

```cpp
#include <smollnet.hpp>

#include <fmt/core.h>

using namespace smollnet;

int main() {
  Tensor input = rand({1, 128}, DataType::f32, Device::CUDA);
  Tensor targets = rand({1, 1}, DataType::f32, Device::CUDA);
  auto targets_h = targets.cpu();

  auto net = Dense(Linear(128, 64), ReLU(), Linear(64, 1));

  for (int epoch = 0; epoch < 32; ++epoch) {
    auto res = net.forward(input);
    auto loss = mse(res, targets);
    fmt::print("[{}] predicted:{} target:{} loss:{}\n", epoch,
               static_cast<float *>(res.cpu().data())[0],
               static_cast<float *>(targets_h.data())[0],
               static_cast<float *>(loss.cpu().data())[0]);
    loss.backward();

    auto optim = SGD(net.parameters(), 0.00001f);
    optim.step();
    optim.zero_grad();
  }
}

```
