# SmollNet

SmollNet is a small deep learning library written in pure CUDA/C++


Example usage:

```cpp
    auto input = smollnet::rand({1,32});

    // fully connected neural net
    auto model = smollnet::Dense(
        smollnet::Linear(32, 16),
        smollnet::ReLU(),
        smollnet::Linear(16,1)
    );

    auto res = model.forward(input);
    res.backward();
```
