#include "tensor.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace smollnet {

Storage::~Storage() {
  if (--refcount == 0) {
    if (device == Device::CUDA)
      cudaFree(ptr);
    else
      free(ptr);
  }
}

Tensor Tensor::sum(int64_t dim) { return ::smollnet::sum(*this, dim); }

Tensor Tensor::add(Tensor &other) {
  ASSERT(p_->dtype == other.impl()->dtype,
         fmt::format("{} vs {}\n", get_name(p_->dtype),
                     get_name(other.impl()->dtype))
             .c_str());
  ASSERT(p_->sizes == other.impl()->sizes, fmt::format("{} vs {}\n", p_->sizes, other.impl()->sizes).c_str());
  ASSERT(p_->elems == other.impl()->elems, fmt::format("{} vs {}\n", p_->elems, other.impl()->elems).c_str());
  ASSERT(p_->ndim == other.impl()->ndim, fmt::format("{} vs {}\n", p_->ndim, other.impl()->ndim).c_str());

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

Tensor Tensor::cuda() {
  if (this->device() == Device::CUDA) {
    return Tensor(*this);
  } else {
    Tensor new_tensor =
        empty(this->dims().data(), this->ndims(), this->dtype(), Device::CUDA);

    CHECK_CUDA(cudaMemcpy(new_tensor.data(), this->data(),
                          this->numel() * element_size(this->dtype()),
                          cudaMemcpyHostToDevice));

    return new_tensor;
  }
}

Tensor Tensor::cpu() {
  if (this->device() == Device::CPU) {
    return Tensor(*this);
  } else {
    Tensor new_tensor =
        empty(this->dims().data(), this->ndims(), this->dtype(), Device::CPU);

    CHECK_CUDA(cudaMemcpy(new_tensor.data(), this->data(),
                          this->numel() * element_size(this->dtype()),
                          cudaMemcpyDeviceToHost));

    return new_tensor;
  }
}

Tensor matmul(Tensor &l, Tensor &r) {
  // Check dims
  ASSERT(l.dims().size() == r.dims().size(),
         fmt::format("{} vs {}\n", l.dims().size(), r.dims().size()).c_str());
  assert(l.dims()[1] == r.dims()[0]);

  Tensor new_tensor = empty({l.dims()[0], r.dims()[1]}, l.dtype(), l.device());

  launch_matmul(new_tensor.data(), l.data(), r.data(), l.dims().data(),
                r.dims().data(), new_tensor.numel());

  return new_tensor;
}

Tensor relu(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device());

  launch_relu(new_tensor.data(), t.data(), t.numel());

  return new_tensor;
}

Tensor tanh(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device());

  launch_tanh(new_tensor.data(), t.data(), t.numel());

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
  if (d == Device::CUDA) {
    CHECK_CUDA(cudaMalloc(&ptr, bytes));
  } else {
    ptr = static_cast<float *>(malloc(bytes));
  }

  storage->ptr = ptr;
  storage->bytes = bytes;
  storage->device = d;

  auto *tensor = new TensorImpl(dims, rank, data);
  tensor->storage = storage;

  return Tensor{tensor};
}

Tensor zeros(const int64_t *dims, size_t rank, DataType data, Device d) {
  auto tensor = empty(dims, rank, data, d);
  if (d == Device::CUDA) {
    CHECK_CUDA(
        cudaMemset(tensor.data(), 0, element_size(data) * product(dims, rank)));
  } else {
    memset(tensor.data(), 0, element_size(data) * product(dims, rank));
  }
  return tensor;
}

Tensor ones(const int64_t *dims, size_t rank, DataType data, Device d) {
  auto tensor = empty(dims, rank, data, d);

  launch_fill(static_cast<float *>(tensor.data()), tensor.numel(), 1.0f);

  return Tensor{tensor};
}

Tensor rand(const int64_t *dims, size_t rank, DataType data, Device d) {
  auto tensor = empty(dims, rank, data, d);

  launch_random_init(tensor.data(), tensor.numel());

  return Tensor{tensor};
}

} // namespace smollnet
