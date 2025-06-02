#include "tensor.hpp"
#include "autograd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <algorithm>
#include <cassert>
#include <cstring>

#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace smollnet {

template <typename GradF>
void SetupAutograd(Tensor const &l, Tensor const &r, Tensor const &n) {
  if (n.requires_grad()) {
    auto *meta = n.autograd();

    meta->grad_fn = std::make_shared<GradF>(l, r);
    meta->is_leaf = false;
  }
}

template <typename GradF>
void SetupAutograd(Tensor const &n, Tensor const &other) {
  if (n.requires_grad()) {
    auto *meta = n.autograd();

    meta->grad_fn = std::make_shared<GradF>(other);
    meta->is_leaf = false;
  }
}

/*
  STORAGE
*/

Storage::~Storage() {

  if (device == Device::CUDA)
    cudaFree(ptr);
  else
    free(ptr);
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

Tensor::Tensor() : impl_(nullptr) {}
Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

bool Tensor::initialized() const noexcept { return impl_ != nullptr; }

TensorImpl *Tensor::impl() const noexcept {
  ASSERT(impl_, "Trying to use uninitialized Tensor!");
  return impl_.get();
}

void Tensor::backward(const Tensor &grad_output) {
  ::smollnet::backward(*this, grad_output);
}

void Tensor::zero_grad() const {
  ASSERT(autograd(), "Tensor doesn't have gradient!");
  ASSERT(grad().initialized(), "Gradient is not initialized!");

  launch_fill(static_cast<float *>(grad().data()), grad().numel(), 0.0f);
}

bool Tensor::requires_grad() const noexcept { return impl()->requires_grad; }

Tensor Tensor::grad() const noexcept {
  ASSERT(impl()->grad, "Accessing uninitialized gradient!");

  return impl()->grad->grad;
}

AutogradMeta *Tensor::autograd() const noexcept { return impl()->grad.get(); }

int64_t Tensor::size(int d) const noexcept { return impl()->sizes[d]; }

int64_t Tensor::ndims() const noexcept { return impl()->ndim; }

Device Tensor::device() const noexcept { return impl()->storage->device; }

DataType Tensor::dtype() const noexcept { return impl()->dtype; }

void *Tensor::data() const noexcept {
  return static_cast<char *>(impl_->storage->ptr);
}

size_t Tensor::numel() const noexcept { return impl_->elems; }

std::array<int64_t, 3> Tensor::dims() const noexcept { return impl_->sizes; }

void Tensor::print() const noexcept {
  auto &t = *impl();
  printf("Tensor: [Refcount: %ld Rank: %ld dim(%ld, %ld, %ld) strides(%ld, "
         "%ld, %ld) "
         "dtype:%s requires_grad:%s]\n\t Storage [Refcount: %ld addr: %p]\n",
         impl_.use_count(), t.ndim, t.sizes[0], t.sizes[1], t.sizes[2],
         t.strides[0], t.strides[1], t.strides[2], get_name(t.dtype),
         requires_grad() ? "true" : "false", t.storage.use_count(),
         t.storage->ptr);
}

void Tensor::print_elms() const noexcept {
  // Could be expensive
  auto *t = cpu().impl();

  fmt::print("Tensor: [");
  // For now we print as contig memory, we can do pretty printing later
  for (int i = 0; i < t->elems; ++i) {
    fmt::print("{}, ", static_cast<float *>(t->storage->ptr)[i]);
  }

  fmt::print("]\n");
}

size_t Tensor::total_bytes() const noexcept {
  return element_size(dtype()) * numel();
}

Tensor Tensor::sum(int64_t dim) const { return ::smollnet::sum(*this, dim); }

Tensor Tensor::matmul(Tensor const &other) const {
  return ::smollnet::matmul(*this, other);
}

Tensor Tensor::add(const Tensor &other) const {

  std::array<int64_t, 3> out_sz = {0, 0, 0};
  bool expand_me = false;
  bool expand_other = false;

  int64_t out_rank = 0;
  for (int i = 0; i < 3; ++i) {
    const auto my_size = size(i);
    const auto other_size = other.size(i);
    ASSERT(my_size == other_size or (my_size == 1 or my_size == 0) or
               (other_size == 1 or other_size == 0),
           fmt::format("Unable to add non-broadcastable Tensors! [{},{},{}] "
                       "and [{},{},{}]",
                       size(0), size(1), size(2), other.size(0), other.size(1),
                       other.size(2)));

    out_sz[i] = std::max(impl()->sizes[i], other.impl()->sizes[i]);

    if (out_sz[i] > 0) {
      out_rank++;
    }

    expand_me |= out_sz[i] != my_size;
    expand_other |= out_sz[i] != other_size;
  }

  auto me_alias = expand_me ? expand(out_sz) : *this;
  auto other_alias = expand_other ? other.expand(out_sz) : other;

  Tensor out = empty(out_sz.data(), out_rank, dtype(), device(),
                     requires_grad() || other.requires_grad());

  if (!expand_me and !expand_other) {
    launch_add(static_cast<float *>(out.data()), static_cast<float *>(data()),
               static_cast<float *>(other.data()), out.numel());
  } else {
    StrideInfo s{};
    s.rank = out_rank;
    for (int i = 0; i < s.rank; ++i) {
      s.size[i] = out_sz[i];
      s.astr[i] = me_alias.impl()->strides[i] / sizeof(float); // bytes → floats
      s.bstr[i] = other_alias.impl()->strides[i] / sizeof(float);
    }

    launch_add_strided(out.data(), me_alias.data(), other_alias.data(), s,
                       out.numel());
  }

  SetupAutograd<AddFunction>(*this, other, out);
  return out;
}

Tensor Tensor::sub(Tensor const &other) const {
  auto &t = *impl();

  assert(t.dtype == other.impl()->dtype);
  assert(t.sizes == other.impl()->sizes);
  assert(t.elems == other.impl()->elems);
  assert(t.ndim == other.impl()->ndim);

  auto new_tensor = empty(t.sizes.data(), t.ndim, t.dtype, t.storage->device,
                          other.requires_grad() or requires_grad());

  launch_sub(static_cast<float *>(new_tensor.data()),
             static_cast<float *>(data()), static_cast<float *>(other.data()),
             t.elems);

  SetupAutograd<SubFunction>(*this, other, new_tensor);

  return new_tensor;
}

Tensor Tensor::transpose(int d0, int d1) const {
  TensorImpl *src = this->impl();

  auto view = std::make_shared<TensorImpl>(*src);
  std::swap(view->sizes[d0], view->sizes[d1]);
  std::swap(view->strides[d0], view->strides[d1]);

  view->storage = src->storage;

  // Copy autograd metadata for views
  if (src->grad) {
    view->grad = src->grad;
  }

  Tensor return_tensor;
  return_tensor.impl_ = view;

  return return_tensor;
}

Tensor Tensor::expand(const std::array<int64_t, 3> &new_sz) const {
  const auto &old = impl()->sizes;
  const int64_t rank = impl()->ndim;

  // check broadcast-compatibility and build new strides
  std::array<int64_t, 3> ns = old;
  std::array<int64_t, 3> st = impl()->strides;

  size_t elems = 1;
  for (int i = 0; i < rank; ++i) {
    if (old[i] == new_sz[i]) {
      ns[i] = new_sz[i];
    } else {
      ASSERT(old[i] == 1,
             fmt::format("expand: non-broadcastable dim {}", old[i]));
      ns[i] = new_sz[i];
      st[i] = 0;
    }

    elems *= ns[i];
  }

  // make view
  auto v = std::make_shared<TensorImpl>();
  v->sizes = ns;
  v->strides = st;
  v->dtype = dtype();
  v->ndim = rank;
  v->elems = elems;
  v->storage = impl()->storage;
  v->requires_grad = requires_grad();

  // share autograd meta
  if (impl()->grad) {
    v->grad = impl()->grad;
  }


  return Tensor(v);
}

Tensor Tensor::cuda() const {
  if (this->device() == Device::CUDA) {
    return Tensor(*this);
  } else {
    Tensor new_tensor =
        empty(dims().data(), ndims(), dtype(), Device::CUDA, requires_grad());

    if (requires_grad()) {
      new_tensor.impl()->grad = impl()->grad;
    }

    CHECK_CUDA(cudaMemcpy(new_tensor.data(), data(),
                          numel() * element_size(dtype()),
                          cudaMemcpyHostToDevice));

    return new_tensor;
  }
}

Tensor Tensor::cpu() const {
  if (this->device() == Device::CPU) {
    return Tensor(*this);
  } else {
    Tensor new_tensor =
        empty(dims().data(), ndims(), dtype(), Device::CPU, requires_grad());

    if (requires_grad()) {
      new_tensor.impl()->grad = impl()->grad;
    }

    CHECK_CUDA(cudaMemcpy(new_tensor.data(), data(),
                          numel() * element_size(dtype()),
                          cudaMemcpyDeviceToHost));

    return new_tensor;
  }
}

Tensor Tensor::copy() const {
  auto new_tensor =
      empty(dims().data(), ndims(), dtype(), device(), requires_grad());

  if (device() == Device::CUDA) {
    CHECK_CUDA(cudaMemcpy(new_tensor.data(), data(),
                          numel() * element_size(dtype()),
                          cudaMemcpyDeviceToDevice));
  } else {
    memcpy(new_tensor.data(), data(), numel() * element_size(dtype()));
  }

  return new_tensor;
}

/*
  FREE FUNCTIONS
*/

Tensor matmul(Tensor const &l, Tensor const &r) {
  // Check dims
  ASSERT(l.dims().size() == r.dims().size(),
         fmt::format("{} vs {}", l.dims().size(), r.dims().size()));
  ASSERT(l.dims()[1] == r.dims()[0],
         fmt::format("{} not equal to {}", l.dims()[1], r.dims()[0]));
  ASSERT(l.device() == r.device(),
         fmt::format("Device mismatch! {} and {}", get_device_name(l.device()),
                     get_device_name(r.device())));

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

  SetupAutograd<ReLUFunction>(new_tensor, t);

  return new_tensor;
}

Tensor tanh(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_tanh(new_tensor.data(), t.data(), t.numel());
  SetupAutograd<TanhFunction>(new_tensor, t);

  return new_tensor;
}

Tensor sigmoid(Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_sigmoid(new_tensor.data(), t.data(), t.numel());
  SetupAutograd<SigmoidFunction>(new_tensor, t);

  return new_tensor;
}

Tensor sum(Tensor const &t, int64_t dim) {
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

Tensor mse(Tensor const &pred, Tensor const &target) {
  ASSERT(pred.dims() == target.dims(), "");

  bool requires_grad = any_requires_grad({pred, target});
  auto new_tensor = zeros({pred.size(0), pred.size(1)}, pred.dtype(),
                          pred.device(), requires_grad);
  launch_mse(new_tensor.data(), pred.data(), target.data(), pred.numel());

  SetupAutograd<MseFunction>(pred, target, new_tensor);

  return new_tensor;
}

Tensor operator+(const Tensor &l, const Tensor &r) { return l.add(r); }

Tensor operator-(const Tensor &l, const Tensor &r) { return l.sub(r); }

Tensor operator*(const Tensor &l, const Tensor &r) { return l.matmul(r); }

Tensor empty(const int64_t *dims, size_t rank, DataType data, Device d,
             bool requires_grad) {
  auto storage = std::make_shared<Storage>();

  float *ptr;
  size_t bytes = element_size(data) * product(dims, rank);
  if (d == Device::CUDA) {
    CHECK_CUDA(cudaMalloc(&ptr, bytes));
  } else {
    ptr = static_cast<float *>(malloc(bytes));
  }

  storage->ptr = ptr;
  storage->device = d;

  auto impl = std::make_shared<TensorImpl>(dims, rank, data);
  impl->storage = storage;
  impl->requires_grad = requires_grad;
  if (requires_grad) {
    impl->grad = std::make_shared<AutogradMeta>();
  }

  return Tensor(impl);
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
