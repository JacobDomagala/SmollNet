#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <array>

namespace smollnet {

using Layout = int32_t;
struct AutogradMeta;

struct Storage {
  void *ptr = nullptr;
  size_t bytes;
  Device device;
  int refcount = 1;

  ~Storage();
};

struct TensorImpl {
  Storage *storage = nullptr;
  std::array<int64_t, 3> sizes = {0, 0, 0};
  std::array<int64_t, 3> strides = {0, 0, 0};
  size_t elems = 1;
  int64_t ndim;
  DataType dtype;
  Layout layout;
  int refcount = 0;

  bool requires_grad = false;
  AutogradMeta *grad = nullptr;

  TensorImpl(const int64_t *dims, int64_t rank, DataType type);
};

class Tensor {
  TensorImpl *p_ = nullptr;

public:
  Tensor();
  explicit Tensor(TensorImpl *p);

  Tensor &operator=(const Tensor &o) noexcept;
  Tensor &operator=(Tensor &&o) noexcept;

  Tensor(const Tensor &o);
  Tensor(Tensor &&o);

  ~Tensor();

  bool initialized() const noexcept;
  TensorImpl *impl() const noexcept;

  void backward(const Tensor& grad_output = Tensor());
  void zero_grad();
  bool requires_grad() const noexcept;
  Tensor grad() const noexcept;
  AutogradMeta* autograd() const noexcept;
  int64_t size(int d) const noexcept;
  int64_t ndims() const noexcept;
  Device device() const noexcept;
  DataType dtype() const noexcept;
  void *data() const noexcept;
  size_t numel() const noexcept;
  std::array<int64_t, 3> dims() const noexcept;
  void print() const noexcept;

  Tensor add(Tensor &other);
  Tensor sub(Tensor &other);
  Tensor sum(int64_t dim);
  Tensor matmul(Tensor& other);

  Tensor transpose(int d0, int d1) const;

  Tensor cuda();
  Tensor cpu();
};


/*
  FREE FUNCTIONS
*/

// Activation functions
Tensor relu(Tensor &t);
Tensor tanh(Tensor &t);
Tensor sigmoid(Tensor &t);

// Operation functions
Tensor matmul(Tensor &l, Tensor &r);
Tensor sum(Tensor &t, int64_t dim);
Tensor operator+(Tensor &l, Tensor &r);
Tensor operator-(Tensor &l, Tensor &r);
Tensor operator*(Tensor &l, Tensor &r);

// Create functions
Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d, bool requires_grad = false);
Tensor zeros(const int64_t *dims, size_t rank, DataType t, Device d, bool requires_grad = false);
Tensor ones(const int64_t *dims, size_t rank, DataType t, Device d, bool requires_grad = false);
Tensor rand(const int64_t *dims, size_t rank, DataType t, Device d, bool requires_grad = false);

template <size_t N>
Tensor empty(const int64_t (&dims)[N], DataType t, Device d, bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return empty(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor zeros(const int64_t (&dims)[N], DataType t, Device d, bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return zeros(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor ones(const int64_t (&dims)[N], DataType t, Device d, bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return ones(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor rand(const int64_t (&dims)[N], DataType t, Device d, bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return rand(dims, N, t, d, requires_grad);
}

} // namespace smollnet
