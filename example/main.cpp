#include <smollnet.hpp>

#include <fmt/core.h>

using namespace smollnet;

int main() {
  constexpr int batch_size = 1024;
  constexpr int num_features = 2048;

  manual_seed(1234);
  Tensor input = rand({batch_size, num_features}, DataType::f32, Device::CUDA);
 // input.print_elms();
  
  Tensor result = mse(input,input);
  // auto targets_h = targets.cpu();

  // auto net = Dense(Linear(num_features, 64), LayerNorm(), GeLU(), Linear(64, 1));

  // for (int epoch = 0; epoch < 64; ++epoch) {
  //   auto res = net.forward(input);
  //   auto loss = mse(res, targets);
  //   fmt::print("epoch[{}]: Loss={}\n", epoch, static_cast<float*>(loss.cpu().data())[0]);
  //   loss.backward();

  //   auto optim = SGD(net.parameters(), 0.005f);
  //   optim.step();
  //   optim.zero_grad();
  // }
}
