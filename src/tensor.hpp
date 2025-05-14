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
  void *ptr;
  size_t bytes;
  Device device;
  int refcount;

  ~Storage();
};

struct TensorImpl {
  Storage *storage = nullptr;
  std::array<int64_t, 3> sizes = {0,0,0};
  std::array<int64_t, 3> strides = {0,0,0};
  size_t elems = 1;
  int64_t ndim;
  DataType dtype;
  Layout layout;
  int refcount = 1;

  bool requires_grad = false;
  AutogradMeta *grad = nullptr;

  TensorImpl(const int64_t* dims, int64_t rank, DataType d) {
    for(size_t d = 0; d < rank; ++d){
      sizes[d] = dims[d];
      elems *= dims[d];
    }

    ndim = rank;
    dtype = d;
  }
};

class Tensor {
  TensorImpl *p_;

public:
  Tensor() : p_(nullptr) {}
  explicit Tensor(TensorImpl *p) : p_(p) { ++p_->refcount; }
  Tensor(const Tensor &o) : p_(o.p_) { ++p_->refcount; }
  ~Tensor() {
    if (p_ && --p_->refcount == 0)
      delete p_;
  }
  TensorImpl *impl() const noexcept { return p_; }
  int64_t size(int d) const noexcept { return p_->sizes[d]; }
  void *data() const noexcept { return static_cast<char *>(p_->storage->ptr); }
  size_t numel() const noexcept { return p_->elems; }
  void print() const noexcept {
    printf("Tensor: [Rank: %ld dim(%ld, %ld, %ld)] dtype:%s\n", p_->ndim, p_->sizes[0], p_->sizes[1], p_->sizes[2], get_name(p_->dtype));
  }

  Tensor sum(Tensor& other);
  Tensor transpose(int d0, int d1) const;
};

Tensor operator+(Tensor& l, Tensor& r);
Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d);
Tensor zeros(const int64_t *dims, size_t rank, DataType t, Device d);
Tensor ones(const int64_t *dims, size_t rank, DataType t, Device d);

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

} // namespace smollnet
