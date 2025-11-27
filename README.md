<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wiki/JacobDomagala/SmollNet/new_logo_light.svg">
  <img alt="logo" src="https://raw.githubusercontent.com/wiki/JacobDomagala/SmollNet/new_logo_dark.svg" width="50%" height="50%">
</picture>

</div>

---

# SmollNet

SmollNet is a small deep learning library written in CUDA/C++ with Python
bindings. It is a learning-oriented neural-network stack: tensors, CUDA kernels,
autograd, basic neural-network layers, optimization, and simple dataset loading
live in one compact codebase.

The project is intentionally small. It is useful for experimenting with how deep
learning libraries are built, writing CUDA kernels, and training simple models
without hiding the machinery behind a large framework.

## Features

- Tensor operations on CPU and CUDA devices
- Autograd for basic arithmetic, matrix multiplication, reductions, activations,
  layer normalization, and mean-squared error
- Neural-network modules: `Linear`, `ReLU`, `GeLU`, `LayerNorm`, and `Dense`
- SGD optimizer
- Python bindings via pybind11
- CSV dataset loading with mini-batch iteration
- One-hot encoding for categorical CSV feature columns

## Requirements

- CMake 3.27+
- Ninja
- Conan 2.x
- CUDA toolkit
- A CUDA-capable GPU for neural-network training examples

## Build

```bash
./build.sh
```

The build script installs the C++ library and Python extension into
`build/smollnet`, then builds the C++ example target.

To import the Python extension from the local build:

```bash
PYTHONPATH=build python3 -c "import smollnet; print(smollnet)"
```

## Python Quick Start

```python
import smollnet

batch_size = 32
input_features = 10

x = smollnet.rand(batch_size, input_features, requires_grad=True)
y = smollnet.rand(batch_size, 1, requires_grad=True)

network = smollnet.Dense(
    smollnet.Linear(input_features, 64),
    smollnet.GeLU(),
    smollnet.Linear(64, 1),
)

optimizer = smollnet.sgd(network.parameters(), lr=0.005)

prediction = network.forward(x)
loss = smollnet.mse(prediction, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Loading CSV Data

CSV files load into a `TensorDataset`, where the final `target_columns` columns
become targets and all preceding columns become inputs. Numeric feature columns
are parsed as floats. Categorical feature columns can be marked by zero-based
CSV column index and are one-hot encoded.

```python
options = smollnet.CSVLoaderOptions()
options.has_header = False
options.target_columns = 1
options.categorical_columns = [0]
options.device = smollnet.Device.CUDA

dataset = smollnet.load_csv_dataset("data/abalone.data", options)

loader_options = smollnet.DataLoaderOptions()
loader_options.batch_size = 128
loader_options.shuffle = True

loader = smollnet.DataLoader(dataset, loader_options)

for batch in loader:
    prediction = network.forward(batch.inputs)
    loss = smollnet.mse(prediction, batch.targets)
```

For numeric-only CSV files, leave `categorical_columns` empty.

## Regression Example

The repository includes the UCI Abalone dataset in `data/abalone.data`. The
example trains a small regression model to predict shell rings from one
categorical feature and seven numeric features:

```bash
PYTHONPATH=build python3 example/abalone_regression.py
```

The current neural-network layers allocate CUDA tensors, so this example needs a
CUDA-capable GPU.

## C++ Usage

```cpp
#include <smollnet.hpp>

using namespace smollnet;

int main() {
  Tensor x = rand({32, 10}, DataType::f32, Device::CUDA, true);
  Tensor y = rand({32, 1}, DataType::f32, Device::CUDA, true);

  auto network = Dense(Linear(10, 64), GeLU(), Linear(64, 1));
  auto optimizer = SGD(network.parameters(), 0.005f);

  auto prediction = network.forward(x);
  auto loss = mse(prediction, y);
  loss.backward();

  optimizer.step();
  optimizer.zero_grad();
}
```

## Current Limitations

- Training examples require CUDA because the neural-network layers currently
  allocate CUDA tensors.
- CSV targets are numeric. String/categorical support is for input feature
  columns.
- The library currently focuses on regression-style examples with `mse`; common
  classification losses such as cross-entropy are not implemented yet.
- Model serialization is not implemented yet.
