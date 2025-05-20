#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <array>

namespace smollnet {

using Layout = int32_t;
using AutogradMeta = int32_t;

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

  TensorImpl(const int64_t *dims, int64_t rank, DataType type) {
    for (size_t d = 0; d < rank; ++d) {
      sizes[d] = dims[d];
      elems *= dims[d];
    }

    if (rank > 0) {
      strides[rank - 1] = element_size(type);
      for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
      }
    }

    ndim = rank;
    dtype = type;
  }
};

class Tensor {
  TensorImpl *p_;

public:
  Tensor() : p_(nullptr) {}

  explicit Tensor(TensorImpl *p) : p_(p) { ++p_->refcount; }

  Tensor &operator=(const Tensor &o) {
    if (this != &o) {
      if (p_ && --p_->refcount == 0)
        delete p_;
      p_ = o.p_;
      if (p_)
        ++p_->refcount;
    }
    return *this;
  }

  Tensor &operator=(Tensor &&o) noexcept {
    if (this != &o) {
      if (p_ && --p_->refcount == 0)
        delete p_;
      p_ = o.p_;
      if (p_)
        ++p_->refcount;
    }
    return *this;
  }

  Tensor(const Tensor &o) : p_(o.p_) { ++p_->refcount; }
  Tensor(Tensor &&o) : p_(o.p_) { ++p_->refcount; }

  ~Tensor() {
    if (p_ && --p_->refcount == 0)
      delete p_;
  }

  TensorImpl *impl() const noexcept { return p_; }

  int64_t size(int d) const noexcept { return p_->sizes[d]; }

  int64_t ndims() const noexcept { return p_->ndim; }

  Device device() const noexcept { return p_->storage->device; }

  DataType dtype() const noexcept { return p_->dtype; }

  void *data() const noexcept { return static_cast<char *>(p_->storage->ptr); }

  size_t numel() const noexcept { return p_->elems; }

  std::array<int64_t, 3> dims() const noexcept { return p_->sizes; }

  void print() const noexcept {
    printf("Tensor: [Refcount: %d Rank: %ld dim(%ld, %ld, %ld) strides(%ld, "
           "%ld, %ld) "
           "dtype:%s]\n\t Storage [Refcount: %d addr: %p]\n",
           p_->refcount, p_->ndim, p_->sizes[0], p_->sizes[1], p_->sizes[2],
           p_->strides[0], p_->strides[1], p_->strides[2], get_name(p_->dtype),
           p_->storage->refcount, p_->storage->ptr);
  }

  Tensor add(Tensor &other);

  Tensor sub(Tensor &other);

  Tensor sum(int64_t dim);

  Tensor transpose(int d0, int d1) const;

  Tensor cuda();

  Tensor cpu();
};

Tensor relu(Tensor &t);
Tensor tanh(Tensor &t);
Tensor sigmoid(Tensor &t);
Tensor matmul(Tensor &l, Tensor &r);

Tensor sum(Tensor &t, int64_t dim);
Tensor operator+(Tensor &l, Tensor &r);
Tensor operator-(Tensor &l, Tensor &r);

Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d);
Tensor zeros(const int64_t *dims, size_t rank, DataType t, Device d);
Tensor ones(const int64_t *dims, size_t rank, DataType t, Device d);
Tensor rand(const int64_t *dims, size_t rank, DataType t, Device d);

template <size_t N>
Tensor empty(const int64_t (&dims)[N], DataType t, Device d) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return empty(dims, N, t, d);
}

template <size_t N>
Tensor zeros(const int64_t (&dims)[N], DataType t, Device d) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return zeros(dims, N, t, d);
}

template <size_t N>
Tensor ones(const int64_t (&dims)[N], DataType t, Device d) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return ones(dims, N, t, d);
}

template <size_t N>
Tensor rand(const int64_t (&dims)[N], DataType t, Device d) {
  static_assert(N <= 3, "We don't support more than 3 dimensional Tensors");
  return rand(dims, N, t, d);
}

} // namespace smollnet
