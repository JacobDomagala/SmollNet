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
void SetupAutograd(const Tensor &l, const Tensor &r, const Tensor &n) {
  if (n.requires_grad()) {
    auto *meta = n.autograd();

    meta->grad_fn = std::make_shared<GradF>(l, r);
    meta->is_leaf = false;
  }
}

template <typename GradF>
void SetupAutograd(const Tensor &n, const Tensor &other) {
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
    strides[rank - 1] = 1;
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
bool Tensor::expanded() const noexcept { return impl_->expanded; }

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
  auto grad_ptr = impl()->grad;
  ASSERT(grad_ptr, "Accessing uninitialized gradient!");

  return grad_ptr->grad;
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

const std::array<int64_t, 3> &Tensor::dims() const noexcept {
  return impl_->sizes;
}

const std::array<int64_t, 3> &Tensor::strides() const noexcept {
  return impl_->strides;
}

void Tensor::print() const {
  if (!initialized()) {
    fmt::print("Uninitialized Tensor\n");
  } else {
    auto &t = *impl();
    fmt::print(
        "Tensor: [Refcount: {} addr: {} Rank: {} dim({}, {}, {}) "
        "strides({}, {}, {}) "
        "dtype:{} requires_grad:{}]\n\t Storage [Refcount: {} addr: {}]\n",
        impl_.use_count(), fmt::ptr(impl_.get()), t.ndim, t.sizes[0],
        t.sizes[1], t.sizes[2], t.strides[0], t.strides[1], t.strides[2],
        get_name(t.dtype), requires_grad(), t.storage.use_count(),
        t.storage->ptr);
  }
}

void Tensor::print_elms() const { fmt::print("{}", to_string()); }

std::string Tensor::to_string() const {

  if (!initialized()) {
    return "[]";
  }

  ASSERT(ndims() <= 3,
         fmt::format("Tensor::print_elms unsupported ndims=={}", ndims()));

  // Could be expensive
  auto t = cpu();
  const float *raw_data = static_cast<const float *>(t.data());

  const auto &sizes = dims();
  const auto &stride = strides();

  fmt::memory_buffer out;

  if (ndims() == 1) {
    fmt::format_to(std::back_inserter(out), "Tensor: ([");
    for (int64_t i = 0; i < sizes[0]; ++i) {
      fmt::format_to(std::back_inserter(out), "{:.4f}{}",
                     raw_data[i * stride[0]], i == sizes[0] - 1 ? "" : ",  ");
    }
    fmt::format_to(std::back_inserter(out), "])\n");
  } else if (ndims() == 2) {
    fmt::format_to(std::back_inserter(out), "Tensor: ([");
    for (int64_t i = 0; i < sizes[0]; ++i) {
      fmt::format_to(std::back_inserter(out), "[");
      for (int64_t j = 0; j < sizes[1]; ++j) {
        fmt::format_to(std::back_inserter(out), "{:.4f}{}",
                       raw_data[i * stride[0] + j * stride[1]],
                       j == sizes[1] - 1 ? "" : ",  ");
      }
      fmt::format_to(std::back_inserter(out), "{}",
                     i == sizes[0] - 1 ? "]" : "],\n          ");
    }
    fmt::format_to(std::back_inserter(out), "])\n");
  } else if (ndims() == 3) {
    fmt::format_to(std::back_inserter(out), "Tensor: ([");
    for (int64_t i = 0; i < sizes[0]; ++i) {
      fmt::format_to(std::back_inserter(out), "[");
      for (int64_t j = 0; j < sizes[1]; ++j) {
        fmt::format_to(std::back_inserter(out), "[");
        for (int64_t k = 0; k < sizes[2]; ++k) {
          fmt::format_to(
              std::back_inserter(out), "{:.4f}{}",
              raw_data[k * stride[2] + j * stride[1] + i * stride[0]],
              k == sizes[2] - 1 ? "" : ",  ");
        }
        fmt::format_to(std::back_inserter(out), "{}",
                       j == sizes[1] - 1 ? "]" : "],\n           ");
      }
      fmt::format_to(std::back_inserter(out), "{}",
                     i == sizes[0] - 1 ? "]" : "],\n\n          ");
    }
    fmt::format_to(std::back_inserter(out), "])\n");
  }

  fmt::format_to(std::back_inserter(out), "])\n");

  return fmt::to_string(out);
}

size_t Tensor::total_bytes() const noexcept {
  return element_size(dtype()) * numel();
}

Tensor Tensor::sum(int64_t dim, bool keep_dim) const {
  return ::smollnet::sum(*this, dim, keep_dim);
}

Tensor Tensor::mul(const Tensor &other) const {
  std::array<int64_t, 3> out_sz = {0, 0, 0};
  bool expand_me = false;
  bool expand_other = false;

  int64_t out_rank = 0;
  for (int i = 0; i < 3; ++i) {
    const auto my_size = size(i);
    const auto other_size = other.size(i);
    ASSERT(
        my_size == other_size or (my_size == 1 or my_size == 0) or
            (other_size == 1 or other_size == 0),
        fmt::format("Unable to multiply non-broadcastable Tensors! [{},{},{}] "
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
    launch_mul(static_cast<float *>(out.data()), static_cast<float *>(data()),
               static_cast<float *>(other.data()), out.numel());
  } else {
    StrideInfo s{};
    s.rank = out_rank;
    for (int i = 0; i < s.rank; ++i) {
      s.output_size[i] = out_sz[i];
      s.a_stride[i] = me_alias.impl()->strides[i];
      s.b_stride[i] = other_alias.impl()->strides[i];
    }

    launch_mul_strided(out.data(), me_alias.data(), other_alias.data(), s,
                       out.numel());
  }

  SetupAutograd<MulFunction>(*this, other, out);
  return out;
}

Tensor Tensor::matmul(const Tensor &other) const {
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
      s.output_size[i] = out_sz[i];
      s.a_stride[i] = me_alias.impl()->strides[i];
      s.b_stride[i] = other_alias.impl()->strides[i];
    }

    launch_add_strided(out.data(), me_alias.data(), other_alias.data(), s,
                       out.numel());
  }

  SetupAutograd<AddFunction>(*this, other, out);
  return out;
}

Tensor Tensor::sub(const Tensor &other) const {
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
    launch_sub(static_cast<float *>(out.data()), static_cast<float *>(data()),
               static_cast<float *>(other.data()), out.numel());
  } else {
    StrideInfo s{};
    s.rank = out_rank;
    for (int i = 0; i < s.rank; ++i) {
      s.output_size[i] = out_sz[i];
      s.a_stride[i] = me_alias.impl()->strides[i];
      s.b_stride[i] = other_alias.impl()->strides[i];
    }

    launch_sub_strided(out.data(), me_alias.data(), other_alias.data(), s,
                       out.numel());
  }

  SetupAutograd<SubFunction>(*this, other, out);
  return out;
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
  v->expanded = true;

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

Tensor matmul(const Tensor &l, const Tensor &r) {
  // Check dims
  ASSERT(l.ndims() >= 2 and r.ndims() >= 2,
         fmt::format("Cannot matrix multiply Tensors with fewer dims than 2! "
                     "lhs.ndims()={} rhs.ndims()={}",
                     l.ndims(), r.ndims()));

  // TODO: allow for broadcast
  ASSERT(l.dims().size() == r.dims().size(),
         fmt::format("{} vs {}", l.dims().size(), r.dims().size()));

  if (l.ndims() == 2) {
    ASSERT(l.dims()[1] == r.dims()[0],
           fmt::format("Incorrect matrix size! lhs number of rows ({}) not "
                       "equal to rhs number of cols ({})",
                       l.dims()[1], r.dims()[0]));
  } else {
    ASSERT(l.dims()[2] == r.dims()[1],
           fmt::format("Incorrect matrix size! lhs number of rows ({}) not "
                       "equal to rhs number of cols ({})",
                       l.dims()[2], r.dims()[1]));
  }

  ASSERT(l.device() == r.device(),
         fmt::format("Device mismatch! {} and {}", get_device_name(l.device()),
                     get_device_name(r.device())));

  bool needs_grad = any_requires_grad({l, r});
  Tensor new_tensor =
      empty({l.dims()[0], r.dims()[1]}, l.dtype(), l.device(), needs_grad);

  StrideInfo stride_info;
  stride_info.output_size[0] = new_tensor.size(0);
  stride_info.output_size[1] = new_tensor.size(1);

  const auto &l_strides = l.strides();
  ;
  stride_info.a_stride[0] = l_strides[0];
  stride_info.a_stride[1] = l_strides[1];
  stride_info.a_stride[2] = l_strides[2];

  const auto &r_strides = r.strides();
  ;
  stride_info.b_stride[0] = r_strides[0];
  stride_info.b_stride[1] = r_strides[1];
  stride_info.b_stride[2] = r_strides[2];

  stride_info.rank = new_tensor.ndims();

  SizeInfo size_info;
  size_info.a_size[0] = l.size(0);
  size_info.a_size[1] = l.size(1);
  size_info.a_size[2] = l.size(2);

  size_info.b_size[0] = r.size(0);
  size_info.b_size[1] = r.size(1);
  size_info.b_size[2] = r.size(2);

  launch_matmul(new_tensor.data(), l.data(), r.data(), stride_info, size_info,
                new_tensor.numel());

  SetupAutograd<MatmulFunction>(l, r, new_tensor);

  return new_tensor;
}

Tensor relu(const Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_relu(new_tensor.data(), t.data(), t.numel());

  SetupAutograd<ReLUFunction>(new_tensor, t);

  return new_tensor;
}

Tensor gelu(const Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_gelu(new_tensor.data(), t.data(), t.numel());

  SetupAutograd<GeLUFunction>(new_tensor, t);
  return new_tensor;
}

Tensor tanh(const Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_tanh(new_tensor.data(), t.data(), t.numel());
  SetupAutograd<TanhFunction>(new_tensor, t);
  return new_tensor;
}

Tensor sigmoid(const Tensor &t) {
  Tensor new_tensor = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                            t.requires_grad());

  launch_sigmoid(new_tensor.data(), t.data(), t.numel());
  SetupAutograd<SigmoidFunction>(new_tensor, t);
  return new_tensor;
}

Tensor sum(const Tensor &t, int64_t dim, bool keep_dim) {
  auto dims = t.dims();
  auto new_rank = keep_dim ? t.ndims() : t.ndims() - 1;
  auto data_type = t.dtype();
  auto device = t.device();

  ASSERT(dim < t.ndims(),
         fmt::format(
             "Tensor sum(tensor,dim,keep_dim): invalid dim={} t.ndims()={}",
             dim, t.ndims()));

  // build output shape
  int64_t out_dims[3] = {0, 0, 0};
  for (int64_t i = 0, o = 0; i < t.ndims(); ++i) {
    if (i != dim) {
      out_dims[o++] = dims[i];
    } else if (keep_dim) {
      out_dims[o++] = 1;
    }
  }

  Tensor new_tensor =
      zeros(out_dims, new_rank, data_type, device, t.requires_grad());

  auto *srcp = t.data();
  auto *dst = new_tensor.data();
  dims[0] = std::max(dims[0], 1l);
  dims[1] = std::max(dims[1], 1l);
  dims[2] = std::max(dims[2], 1l);

  StrideAndSize s_input;
  s_input.size = dims;
  s_input.stride = t.strides();
  s_input.rank = t.ndims();

  StrideAndSize s_output;
  s_output.size = new_tensor.dims();
  s_output.stride = new_tensor.strides();
  s_output.rank = new_tensor.ndims();

  if (dim == 0) {
    launch_sum_dim0(dst, srcp, s_input, s_output);
  } else if (dim == 1) {
    launch_sum_dim1(dst, srcp, s_input, s_output);
  } else {
    // dim==2
    launch_sum_dim2(dst, srcp, s_input, s_output);
  }

  return new_tensor;
}

Tensor mul(const Tensor &left, const Tensor &right) { return left.mul(right); }

Tensor mse(const Tensor &pred, const Tensor &target) {
  ASSERT(pred.dims() == target.dims(), "");

  bool requires_grad = any_requires_grad({pred, target});
  auto new_tensor = zeros({1}, pred.dtype(), pred.device(), requires_grad);
  launch_mse(new_tensor.data(), pred.data(), target.data(), pred.numel());

  SetupAutograd<MseFunction>(pred, target, new_tensor);
  return new_tensor;
}

Tensor operator+(const Tensor &l, const Tensor &r) { return l.add(r); }

Tensor operator-(const Tensor &l, const Tensor &r) { return l.sub(r); }

Tensor operator*(const Tensor &l, const Tensor &r) { return l.mul(r); }

Tensor &operator+=(Tensor &l, const Tensor &r) {
  l = l + r;
  return l;
}

Tensor &operator-=(Tensor &l, const Tensor &r) {
  l = l - r;
  return l;
}

Tensor &operator*=(Tensor &l, const Tensor &r) {
  l = l * r;
  return l;
}

Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad) {
  auto storage = std::make_shared<Storage>();

  float *ptr;
  size_t bytes = element_size(t) * product(dims, rank);
  if (d == Device::CUDA) {
    CHECK_CUDA(cudaMalloc(&ptr, bytes));
  } else {
    ptr = static_cast<float *>(malloc(bytes));
  }

  storage->ptr = ptr;
  storage->device = d;

  auto impl = std::make_shared<TensorImpl>(dims, rank, t);
  impl->storage = storage;
  impl->requires_grad = requires_grad;
  if (requires_grad) {
    impl->grad = std::make_shared<AutogradMeta>();
  }

  return Tensor(impl);
}

Tensor zeros(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad) {
  auto tensor = empty(dims, rank, t, d, requires_grad);
  if (d == Device::CUDA) {
    CHECK_CUDA(
        cudaMemset(tensor.data(), 0, element_size(t) * product(dims, rank)));
  } else {
    memset(tensor.data(), 0, element_size(t) * product(dims, rank));
  }
  return tensor;
}

Tensor ones(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad) {
  auto tensor = empty(dims, rank, t, d, requires_grad);

  launch_fill(static_cast<float *>(tensor.data()), tensor.numel(), 1.0f);

  return Tensor{tensor};
}

Tensor rand(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad) {
  auto tensor = empty(dims, rank, t, d, requires_grad);

  launch_random_init(tensor.data(), tensor.numel());

  return Tensor{tensor};
}

} // namespace smollnet
