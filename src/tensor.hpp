#pragma once

#include "types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace smollnet {

struct AutogradMeta;

struct Storage {
  void *ptr = nullptr;
  Device device;

  ~Storage();
};

struct TensorImpl {
  std::shared_ptr<Storage> storage = nullptr;
  std::array<int64_t, 3> sizes = {0, 0, 0};
  std::array<int64_t, 3> strides = {0, 0, 0};

  size_t elems = 1;
  int64_t ndim;

  DataType dtype;

  bool requires_grad = false;
  std::shared_ptr<AutogradMeta> grad = nullptr;

  TensorImpl() = default;
  TensorImpl(const TensorImpl &) = default;
  TensorImpl(TensorImpl &&) = default;
  TensorImpl &operator=(const TensorImpl &) = default;
  TensorImpl &operator=(TensorImpl &&) = default;
  ~TensorImpl() = default;

  TensorImpl(const int64_t *dims, int64_t rank, DataType type);
};

class Tensor {
  std::shared_ptr<TensorImpl> impl_ = nullptr;

public:
  Tensor();
  Tensor(std::shared_ptr<TensorImpl> impl);

  Tensor &operator=(const Tensor &o) noexcept = default;
  Tensor &operator=(Tensor &&o) noexcept = default;

  Tensor(const Tensor &o) = default;
  Tensor(Tensor &&o) = default;

  ~Tensor() = default;

  TensorImpl *impl() const noexcept;
  bool initialized() const noexcept;
  void backward(const Tensor &grad_output = Tensor());
  void zero_grad() const;
  bool requires_grad() const noexcept;
  Tensor grad() const noexcept;
  AutogradMeta *autograd() const noexcept;
  int64_t size(int d) const noexcept;
  int64_t ndims() const noexcept;
  Device device() const noexcept;
  DataType dtype() const noexcept;
  void *data() const noexcept;
  size_t numel() const noexcept;
  std::array<int64_t, 3> dims() const noexcept;
  void print() const noexcept;
  void print_elms() const noexcept;
  size_t total_bytes() const noexcept;

  Tensor add(Tensor const &other) const;
  Tensor sub(Tensor const &other) const;
  Tensor sum(int64_t dim) const;
  Tensor matmul(Tensor const &other) const;

  Tensor transpose(int d0, int d1) const;
  Tensor expand(const std::array<int64_t, 3> &new_dims) const;

  Tensor cuda() const;
  Tensor cpu() const;
  Tensor copy() const;
};

/*
  FREE FUNCTIONS
*/

// Activation functions
Tensor relu(Tensor &t);
Tensor tanh(Tensor &t);
Tensor sigmoid(Tensor &t);

// Operation functions
Tensor matmul(Tensor const &l, Tensor const &r);
Tensor sum(Tensor const &t, int64_t dim);
Tensor operator+(Tensor const &l, Tensor const &r);
Tensor operator-(Tensor const &l, Tensor const &r);
Tensor operator*(Tensor const &l, Tensor const &r);

Tensor mse(Tensor const &pred, Tensor const &target);

// Create functions
Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad = false);
Tensor zeros(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad = false);
Tensor ones(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad = false);
Tensor rand(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad = false);

template <size_t N>
Tensor empty(const int64_t (&dims)[N], DataType t, Device d,
             bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return empty(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor zeros(const int64_t (&dims)[N], DataType t, Device d,
             bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return zeros(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor ones(const int64_t (&dims)[N], DataType t, Device d,
            bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return ones(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor rand(const int64_t (&dims)[N], DataType t, Device d,
            bool requires_grad = false) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return rand(dims, N, t, d, requires_grad);
}

} // namespace smollnet
