#include "tensor.hpp"
#include "kernels.cuh"

#include <cassert>
#include <cuda_runtime.h>

namespace smollnet {

Storage::~Storage() {
  if (--refcount == 0)
    cudaFree(ptr);
}

Tensor Tensor::sum(int64_t dim) { return ::smollnet::sum(*this, dim); }

Tensor Tensor::add(Tensor &other) {
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

Tensor Tensor::sub(Tensor &other) {
  assert(p_->dtype == other.impl()->dtype);
  assert(p_->sizes == other.impl()->sizes);
  assert(p_->elems == other.impl()->elems);
  assert(p_->ndim == other.impl()->ndim);

  auto new_tensor =
      empty(p_->sizes.data(), p_->ndim, p_->dtype, p_->storage->device);

  launch_sub(static_cast<float *>(new_tensor.data()),
             static_cast<float *>(data()), static_cast<float *>(other.data()),
             p_->elems);

  return new_tensor;
}

Tensor Tensor::transpose(int d0, int d1) const {
  TensorImpl *src = this->impl();
  ++src->storage->refcount;

  TensorImpl *view = new TensorImpl(*src);
  std::swap(view->sizes[d0], view->sizes[d1]);
  std::swap(view->strides[d0], view->strides[d1]);

  view->storage = src->storage;
  view->refcount = 1;

  return Tensor(view);
}

Tensor matmul(Tensor &l, Tensor &r) {
  // Check dims
  assert(l.dims().size() == r.dims().size());
  assert(l.dims()[1] == r.dims()[0]);

  Tensor new_tensor = empty({l.dims()[0], r.dims()[1]}, l.dtype(), l.device());

  launch_matmul(new_tensor.data(), l.data(), r.data(),
                l.dims().data(), r.dims().data(), new_tensor.numel());

  return new_tensor;
}

Tensor sum(Tensor &t, int64_t dim) {
  auto *src = t.impl();
  auto &dims = src->sizes;
  auto new_rank = src->ndim - 1;
  auto data_type = src->dtype;
  auto device = src->storage->device;

  // TODO: Check for correct dim!

  // build output shape
  int64_t out_dims[3];
  for (int64_t i = 0, o = 0; i < src->ndim; ++i)
    if (i != dim)
      out_dims[o++] = src->sizes[i];

  Tensor new_tensor = empty(out_dims, new_rank, data_type, device);

  auto *srcp = t.data();
  auto *dst = new_tensor.data();
  if (dim == 0) {
    int64_t d0 = src->sizes[0];
    int64_t rest = std::max(src->sizes[1] * src->sizes[2], 1l);
    launch_sum_dim0(dst, srcp, d0, rest);
  } else if (dim == 1) {
    launch_sum_dim1(dst, srcp, src->sizes[0], src->sizes[1], src->sizes[2]);
  } else {
    // dim==2
    launch_sum_dim2(dst, srcp, src->sizes[0], src->sizes[1], src->sizes[2]);
  }
  return new_tensor;
}

Tensor operator+(Tensor &l, Tensor &r) { return l.add(r); }
Tensor operator-(Tensor &l, Tensor &r) { return l.sub(r); }

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
