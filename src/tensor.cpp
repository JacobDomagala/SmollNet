#include "tensor.hpp"
#include "autograd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace smollnet {

template <typename GradF> void SetupAutograd(Tensor &l, Tensor &r, Tensor &n) {
  if (l.requires_grad() or r.requires_grad()) {
    auto *meta = n.autograd();

    meta->grad_fn = std::make_shared<GradF>(l, r);
    meta->is_leaf = false;
  }
}
/*
  STORAGE
*/

Storage::~Storage() {
  if (--refcount == 0) {
    if (device == Device::CUDA)
      cudaFree(ptr);
    else
      free(ptr);
  }
}

/*
  TENSORIMPL
*/

TensorImpl::TensorImpl(const int64_t *dims, int64_t rank, DataType type) {
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

/*
  TENSOR
*/

Tensor::Tensor() : p_(nullptr) {}

Tensor::Tensor(TensorImpl *p) : p_(p) {
  if (p_)
    ++p_->refcount;
}

Tensor &Tensor::operator=(const Tensor &o) noexcept {
  if (this != &o) {
    if (p_ && --p_->refcount == 0)
      delete p_;
    p_ = o.p_;
    if (p_)
      ++p_->refcount;
  }
  return *this;
}

Tensor &Tensor::operator=(Tensor &&o) noexcept {
  if (this != &o) {
    if (p_ && --p_->refcount == 0)
      delete p_;
    p_ = o.p_;
    if (p_)
      ++p_->refcount;
  }
  return *this;
}

Tensor::Tensor(const Tensor &o) : p_(o.p_) {
  if (p_)
    ++p_->refcount;
}
Tensor::Tensor(Tensor &&o) : p_(o.p_) {
  if (p_)
    ++p_->refcount;
}

Tensor::~Tensor() {
  if (p_ && --p_->refcount == 0)
    delete p_;
}

bool Tensor::initialized() const noexcept {
  return p_;
}

TensorImpl *Tensor::impl() const noexcept {
  ASSERT(p_, "Trying to use uninitialized Tensor!");
  return p_;
}

void Tensor::backward(const Tensor &grad_output) {
  ::smollnet::backward(*this, grad_output);
}
void Tensor::zero_grad() {
  ASSERT(autograd(), "Tensor doesn't have gradient!");

  launch_fill(static_cast<float *>(grad().data()), grad().numel(), 0.0f);
}
bool Tensor::requires_grad() const noexcept { return impl()->requires_grad; }
Tensor Tensor::grad() const noexcept { return impl()->grad->grad; }
AutogradMeta *Tensor::autograd() const noexcept { return impl()->grad; }
int64_t Tensor::size(int d) const noexcept { return impl()->sizes[d]; }
int64_t Tensor::ndims() const noexcept { return impl()->ndim; }
Device Tensor::device() const noexcept { return impl()->storage->device; }
DataType Tensor::dtype() const noexcept { return impl()->dtype; }

void *Tensor::data() const noexcept {
  return static_cast<char *>(p_->storage->ptr);
}

size_t Tensor::numel() const noexcept { return p_->elems; }

std::array<int64_t, 3> Tensor::dims() const noexcept { return p_->sizes; }

void Tensor::print() const noexcept {
  auto &t = *impl();
  printf("Tensor: [Refcount: %d Rank: %ld dim(%ld, %ld, %ld) strides(%ld, "
         "%ld, %ld) "
         "dtype:%s requires_grad:%s]\n\t Storage [Refcount: %d addr: %p]\n",
         t.refcount, t.ndim, t.sizes[0], t.sizes[1], t.sizes[2], t.strides[0],
         t.strides[1], t.strides[2], get_name(t.dtype),
         requires_grad() ? "true" : "false", t.storage->refcount,
         t.storage->ptr);
}

Tensor Tensor::sum(int64_t dim) { return ::smollnet::sum(*this, dim); }

Tensor Tensor::add(Tensor &other) {
  auto &t = *impl();

  ASSERT(t.dtype == other.impl()->dtype,
         fmt::format("{} vs {}\n", get_name(t.dtype),
                     get_name(other.impl()->dtype))
             .c_str());
  ASSERT(t.sizes == other.impl()->sizes,
         fmt::format("{} vs {}\n", t.sizes, other.impl()->sizes).c_str());
  ASSERT(t.elems == other.impl()->elems,
         fmt::format("{} vs {}\n", t.elems, other.impl()->elems).c_str());
  ASSERT(t.ndim == other.impl()->ndim,
         fmt::format("{} vs {}\n", t.ndim, other.impl()->ndim).c_str());

  auto new_tensor = empty(t.sizes.data(), t.ndim, t.dtype, t.storage->device,
                          other.requires_grad());

  launch_add(static_cast<float *>(new_tensor.data()),
             static_cast<float *>(data()), static_cast<float *>(other.data()),
             t.elems);

  SetupAutograd<AddFunction>(*this, other, new_tensor);

  return new_tensor;
}

Tensor Tensor::sub(Tensor &other) {
  auto &t = *impl();

  assert(t.dtype == other.impl()->dtype);
  assert(t.sizes == other.impl()->sizes);
  assert(t.elems == other.impl()->elems);
  assert(t.ndim == other.impl()->ndim);

  auto new_tensor = empty(t.sizes.data(), t.ndim, t.dtype, t.storage->device,
                          other.requires_grad());

  launch_sub(static_cast<float *>(new_tensor.data()),
             static_cast<float *>(data()), static_cast<float *>(other.data()),
             t.elems);

  SetupAutograd<SubFunction>(*this, other, new_tensor);

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

  // Copy autograd metadata for views
  if (src->grad) {
    view->grad = src->grad;
    ++view->grad->refcount;
  }

  return Tensor(view);
}

Tensor Tensor::cuda() {
  if (this->device() == Device::CUDA) {
    return Tensor(*this);
  } else {
    Tensor new_tensor =
        empty(dims().data(), ndims(), dtype(), Device::CUDA, requires_grad());

    if (requires_grad()) {
      new_tensor.impl()->grad = impl()->grad;
      ++new_tensor.impl()->grad->refcount;
    }

    CHECK_CUDA(cudaMemcpy(new_tensor.data(), data(),
                          numel() * element_size(dtype()),
                          cudaMemcpyHostToDevice));

    return new_tensor;
  }
}

Tensor Tensor::cpu() {
  if (this->device() == Device::CPU) {
    return Tensor(*this);
  } else {
    Tensor new_tensor =
        empty(dims().data(), ndims(), dtype(), Device::CPU, requires_grad());

    if (requires_grad()) {
      new_tensor.impl()->grad = impl()->grad;
      ++new_tensor.impl()->grad->refcount;
    }

    CHECK_CUDA(cudaMemcpy(new_tensor.data(), data(),
                          numel() * element_size(dtype()),
                          cudaMemcpyDeviceToHost));

    return new_tensor;
  }
}

/*
  FREE FUNCTIONS
*/

Tensor matmul(Tensor &l, Tensor &r) {
  // Check dims
  ASSERT(l.dims().size() == r.dims().size(),
         fmt::format("{} vs {}\n", l.dims().size(), r.dims().size()).c_str());
  assert(l.dims()[1] == r.dims()[0]);

  bool needs_grad = any_requires_grad({l, r});
  Tensor new_tensor =
      empty({l.dims()[0], r.dims()[1]}, l.dtype(), l.device(), needs_grad);

  launch_matmul(new_tensor.data(), l.data(), r.data(), l.dims().data(),
                r.dims().data(), new_tensor.numel());

  SetupAutograd<MatmulFunction>(l, r, new_tensor);

  return new_tensor;
}

Tensor relu(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_relu(new_tensor.data(), t.data(), t.numel());

  return new_tensor;
}

Tensor tanh(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_tanh(new_tensor.data(), t.data(), t.numel());

  return new_tensor;
}

Tensor sigmoid(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_sigmoid(new_tensor.data(), t.data(), t.numel());

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

Tensor empty(const int64_t *dims, size_t rank, DataType data, Device d,
             bool requires_grad) {
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
  tensor->requires_grad = requires_grad;
  if (requires_grad) {
    tensor->grad = new AutogradMeta();
  }

  return Tensor{tensor};
}

Tensor zeros(const int64_t *dims, size_t rank, DataType data, Device d,
             bool requires_grad) {
  auto tensor = empty(dims, rank, data, d, requires_grad);
  if (d == Device::CUDA) {
    CHECK_CUDA(
        cudaMemset(tensor.data(), 0, element_size(data) * product(dims, rank)));
  } else {
    memset(tensor.data(), 0, element_size(data) * product(dims, rank));
  }
  return tensor;
}

Tensor ones(const int64_t *dims, size_t rank, DataType data, Device d,
            bool requires_grad) {
  auto tensor = empty(dims, rank, data, d, requires_grad);

  launch_fill(static_cast<float *>(tensor.data()), tensor.numel(), 1.0f);

  return Tensor{tensor};
}

Tensor rand(const int64_t *dims, size_t rank, DataType data, Device d,
            bool requires_grad) {
  auto tensor = empty(dims, rank, data, d, requires_grad);

  launch_random_init(tensor.data(), tensor.numel());

  return Tensor{tensor};
}

} // namespace smollnet
