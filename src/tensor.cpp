#include "tensor.hpp"
#include "kernels.cuh"

#include <cassert>
#include <cuda_runtime.h>

namespace smollnet {

Storage::~Storage() {
  if (--refcount == 0)
    cudaFree(ptr);
}

Tensor Tensor::sum(Tensor &other) {
  assert(p_->dtype == other.impl()->dtype);
  assert(p_->sizes == other.impl()->sizes);
  assert(p_->elems == other.impl()->elems);
  assert(p_->ndim == other.impl()->ndim);

  auto new_tensor =
      empty(p_->sizes.data(), p_->ndim, p_->dtype, p_->storage->device);

  launch_add(static_cast<float *>(new_tensor.data()),
             static_cast<float *>(data()), static_cast<float *>(other.data()),
             p_->elems);

  return new_tensor;
}

Tensor Tensor::transpose(int d0, int d1) const {
  TensorImpl* src = this->impl();
  TensorImpl* view = new TensorImpl(*src);  // shallow copy
  std::swap(view->sizes[d0], view->sizes[d1]);
  std::swap(view->strides[d0], view->strides[d1]);
  ++src->storage->refcount;
  view->storage = src->storage;
  view->refcount = 1;
  return Tensor(view);
}

Tensor operator+(Tensor &l, Tensor &r) { return l.sum(r); }

Tensor empty(const int64_t *dims, size_t rank, DataType data, Device d) {
  auto *storage = new Storage;

  float *ptr;
  size_t bytes = element_size(data) * product(dims, rank);
  cudaMalloc(&ptr, bytes);

  storage->ptr = ptr;
  storage->bytes = bytes;
  storage->device = d;

  auto *tensor = new TensorImpl(dims, rank, data);
  tensor->storage = storage;

  return Tensor{tensor};
}

Tensor zeros(const int64_t *dims, size_t rank, DataType data, Device d) {
  auto tensor = empty(dims, rank, data, d);
  cudaMemset(tensor.data(), 0, element_size(data) * product(dims, rank));

  return tensor;
}

Tensor ones(const int64_t *dims, size_t rank, DataType data, Device d) {
  auto tensor = empty(dims, rank, data, d);

  launch_fill(static_cast<float *>(tensor.data()), tensor.numel(), 1.0f);

  return Tensor{tensor};
}

} // namespace smollnet
