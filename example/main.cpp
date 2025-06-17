#include <smollnet.hpp>

#include <fmt/core.h>

using namespace smollnet;

int main() {
  constexpr int input_size = 10;

  Tensor input = rand({input_size, 128}, DataType::f32, Device::CUDA);
  Tensor targets = rand({input_size, 1}, DataType::f32, Device::CUDA);
  auto targets_h = targets.cpu();

  auto net = Dense(Linear(128, 64), LayerNorm(), GeLU(), Linear(64, 1));

  for (int epoch = 0; epoch < 64; ++epoch) {
    auto res = net.forward(input);
    auto loss = mse(res, targets);

    loss.backward();

    auto optim = SGD(net.parameters(), 0.005f);
    optim.step();
    optim.zero_grad();
  }

  targets.print_elms();

  auto predicted = net.forward(input);
  predicted.print_elms();

  auto loss = mse(predicted, targets);
  loss.print_elms();
}
