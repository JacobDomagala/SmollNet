#pragma once

#include "types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace smollnet {

struct AutogradMeta;

struct Storage {
  void *ptr = nullptr;
  Device device = Device::CPU;

  Storage() = default;
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  Storage(Storage &&) = delete;
  Storage &operator=(Storage &&) = delete;

  ~Storage();
};

struct TensorImpl {
  std::shared_ptr<Storage> storage = nullptr;
  TensorShape sizes = {};
  TensorShape strides = {};

  bool expanded = false;
  size_t elems = 1;
  int64_t ndim = 0;

  DataType dtype = DataType::f32;

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
  explicit Tensor();
  explicit Tensor(std::shared_ptr<TensorImpl> impl);

  Tensor &operator=(const Tensor &o) noexcept = default;
  Tensor &operator=(Tensor &&o) noexcept = default;

  Tensor(const Tensor &o) = default;
  Tensor(Tensor &&o) = default;

  ~Tensor() = default;

  TensorImpl *impl() const noexcept;
  bool initialized() const noexcept;
  bool expanded() const noexcept;
  void backward(const Tensor &grad_output = Tensor());
  void zero_grad() const;
  bool requires_grad() const noexcept;
  Tensor grad() const noexcept;
  AutogradMeta *autograd() const noexcept;
  int64_t size(int64_t d) const noexcept;
  int64_t ndims() const noexcept;
  Device device() const noexcept;
  DataType dtype() const noexcept;
  void *data() const noexcept;
  size_t numel() const noexcept;
  const TensorShape &dims() const noexcept;
  const TensorShape &strides() const noexcept;
  void print() const;
  void print_elms() const;
  std::string to_string() const;
  size_t total_bytes() const noexcept;

  Tensor neg() const;
  Tensor add(const Tensor&other) const;
  Tensor add(float scalar) const;
  Tensor sub(const Tensor&other) const;
  Tensor sub(float scalar) const;
  Tensor rsub(float scalar) const;
  Tensor sum(int64_t dim, bool keep_dim = false) const;
  Tensor mul(const Tensor&other) const;
  Tensor mul(float scalar) const;
  Tensor div(const Tensor&other) const;
  Tensor div(float scalar) const;
  Tensor rdiv(float scalar) const;
  Tensor matmul(const Tensor&other) const;

  Tensor transpose(int d0, int d1) const;
  Tensor expand(const TensorShape &new_sz) const;

  Tensor cuda() const;
  Tensor cpu() const;
  Tensor copy() const;
};

/*
  FREE FUNCTIONS
*/

// Activation functions
Tensor relu(const Tensor &t);
Tensor gelu(const Tensor &t);
Tensor tanh(const Tensor &t);
Tensor sigmoid(const Tensor &t);

// Operation functions
Tensor matmul(const Tensor&l, const Tensor&r);
Tensor neg(const Tensor&t);
Tensor add(const Tensor& left, const Tensor& right);
Tensor sub(const Tensor& left, const Tensor& right);
Tensor mul(const Tensor& left, const Tensor& right);
Tensor div(const Tensor& left, const Tensor& right);
Tensor sum(const Tensor&t, int64_t dim, bool keep_dim = false);
Tensor operator-(const Tensor&t);
Tensor operator+(const Tensor&l, const Tensor&r);
Tensor operator-(const Tensor&l, const Tensor&r);
Tensor operator*(const Tensor&l, const Tensor&r);
Tensor operator/(const Tensor&l, const Tensor&r);
Tensor operator+(const Tensor&l, float scalar);
Tensor operator+(float scalar, const Tensor&r);
Tensor operator-(const Tensor&l, float scalar);
Tensor operator-(float scalar, const Tensor&r);
Tensor operator*(const Tensor&l, float scalar);
Tensor operator*(float scalar, const Tensor&r);
Tensor operator/(const Tensor&l, float scalar);
Tensor operator/(float scalar, const Tensor&r);
Tensor &operator+=(Tensor&l, const Tensor &r);
Tensor &operator-=(Tensor&l, const Tensor &r);
Tensor &operator*=(Tensor&l, const Tensor &r);
Tensor &operator/=(Tensor&l, const Tensor &r);
Tensor &operator+=(Tensor&l, float scalar);
Tensor &operator-=(Tensor&l, float scalar);
Tensor &operator*=(Tensor&l, float scalar);
Tensor &operator/=(Tensor&l, float scalar);

Tensor mse(const Tensor&pred, const Tensor&target);

// Create functions
Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad = false);
Tensor zeros(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad = false);
Tensor ones(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad = false);
Tensor rand(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad = false);
Tensor full_like(const Tensor &t, float value, bool requires_grad = false);
Tensor zeros_like(const Tensor &t, bool requires_grad = false);
Tensor ones_like(const Tensor &t, bool requires_grad = false);
Tensor rand_like(const Tensor &t, bool requires_grad = false);
void manual_seed(unsigned long long seed);

template <size_t N>
Tensor empty(const int64_t (&dims)[N], DataType t, Device d,
             bool requires_grad = false) {
  static_assert(N <= kMaxTensorDims,
                "We don't support more than kMaxTensorDims dimensional Tensors");
  return empty(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor zeros(const int64_t (&dims)[N], DataType t, Device d,
             bool requires_grad = false) {
  static_assert(N <= kMaxTensorDims,
                "We don't support more than kMaxTensorDims dimensional Tensors");
  return zeros(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor ones(const int64_t (&dims)[N], DataType t, Device d,
            bool requires_grad = false) {
  static_assert(N <= kMaxTensorDims,
                "We don't support more than kMaxTensorDims dimensional Tensors");
  return ones(dims, N, t, d, requires_grad);
}

template <size_t N>
Tensor rand(const int64_t (&dims)[N], DataType t, Device d,
            bool requires_grad = false) {
  static_assert(N <= kMaxTensorDims,
                "We don't support more than kMaxTensorDims dimensional Tensors");
  return rand(dims, N, t, d, requires_grad);
}

} // namespace smollnet
