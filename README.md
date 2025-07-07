# SmollNet

![logo](https://raw.githubusercontent.com/wiki/JacobDomagala/SmollNet/logo_256.png)

SmollNet is a small deep learning library written in pure CUDA/C++


Example usage:

```cpp
#include <smollnet.hpp>

#include <fmt/core.h>

using namespace smollnet;

int main() {
  constexpr batch_size = 10;
  Tensor input = rand({batch_size, 128}, DataType::f32, Device::CUDA);
  Tensor targets = rand({batch_size, 1}, DataType::f32, Device::CUDA);
  auto targets_h = targets.cpu();

  auto net = Dense(Linear(128, 64), ReLU(), Linear(64, 1));

  for (int epoch = 0; epoch < 32; ++epoch) {
    auto res = net.forward(input);
    auto loss = mse(res, targets);
    fmt::print("epoch:{} predicted:{} target:{} loss:{}\n", epoch,
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

Or in python:

```python
import smollnet

batch_size = 10
num_features = 128

# Generate random data
x = smollnet.rand(batch_size, num_features, requires_grad=True)
y = smollnet.rand(batch_size, 1, requires_grad=True)

# Create network
network = smollnet.Dense(
    smollnet.Linear(num_features, 64),
    smollnet.GeLU(),
    smollnet.Linear(64, 1))

# Training loop
num_epochs = 64
for epoch in range(num_epochs):
    # Forward pass
    output = network.forward(x)

    # Compute loss
    loss = smollnet.mse(output, y)
    loss.backward()

    # Backward pass
    optimizer = smollnet.sgd(network.parameters(), lr=0.005)

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")
```
