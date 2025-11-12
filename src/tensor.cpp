#include "tensor.hpp"
#include "autograd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <random>
#include <string>

#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace smollnet {

namespace {

std::mt19937 &cpu_random_generator() {
  // NOLINTNEXTLINE
  static std::mt19937 generator(1234U);
  return generator;
}

std::string shape_to_string(const TensorShape &shape, int64_t rank) {
  return fmt::format("[{}]",
                     fmt::join(shape.begin(), shape.begin() + rank, ", "));
}

int64_t infer_rank(const TensorShape &shape) {
  int64_t rank = 0;
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    if (shape[dim] != 0) {
      rank = static_cast<int64_t>(dim + 1);
    }
  }
  return rank;
}

bool is_dense_contiguous(const Tensor &t) {
  int64_t expected_stride = 1;
  for (int64_t dim = t.ndims() - 1; dim >= 0; --dim) {
    if (t.strides()[dim] != expected_stride) {
      return false;
    }
    expected_stride *= t.size(dim);
  }
  return true;
}

Tensor make_broadcast_view(const Tensor &input, const TensorShape &out_shape,
                           int64_t out_rank) {
  ASSERT(out_rank >= input.ndims(),
         fmt::format("Cannot broadcast tensor rank {} to rank {}",
                     input.ndims(), out_rank));

  TensorShape sizes{};
  TensorShape strides{};
  size_t elems = 1;
  bool changed = input.ndims() != out_rank;
  const int64_t rank_offset = out_rank - input.ndims();

  for (int64_t dim = 0; dim < out_rank; ++dim) {
    const int64_t input_dim = dim - rank_offset;
    const int64_t old_size = input_dim >= 0 ? input.size(input_dim) : 1;
    const int64_t old_stride =
        input_dim >= 0 ? input.strides()[input_dim] : 0;

    ASSERT(old_size == out_shape[dim] || old_size == 1,
           fmt::format("Cannot broadcast shape {} to {}",
                       shape_to_string(input.dims(), input.ndims()),
                       shape_to_string(out_shape, out_rank)));

    sizes[dim] = out_shape[dim];
    strides[dim] = old_size == out_shape[dim] ? old_stride : 0;
    elems *= static_cast<size_t>(sizes[dim]);
    changed |= input_dim != dim || old_size != out_shape[dim] ||
               old_stride != strides[dim];
  }

  if (!changed) {
    return input;
  }

  auto view = std::make_shared<TensorImpl>(*input.impl());
  view->sizes = sizes;
  view->strides = strides;
  view->ndim = out_rank;
  view->elems = elems;
  view->expanded = true;
  return Tensor(view);
}

TensorShape broadcast_shape(const Tensor &lhs, const Tensor &rhs,
                            const char *op_name) {
  TensorShape out_shape{};
  const int64_t out_rank = std::max(lhs.ndims(), rhs.ndims());
  const int64_t lhs_offset = out_rank - lhs.ndims();
  const int64_t rhs_offset = out_rank - rhs.ndims();

  for (int64_t dim = 0; dim < out_rank; ++dim) {
    const int64_t lhs_dim = dim - lhs_offset;
    const int64_t rhs_dim = dim - rhs_offset;
    const int64_t lhs_size = lhs_dim >= 0 ? lhs.size(lhs_dim) : 1;
    const int64_t rhs_size = rhs_dim >= 0 ? rhs.size(rhs_dim) : 1;

    ASSERT(lhs_size == rhs_size || lhs_size == 1 || rhs_size == 1,
           fmt::format("Unable to {} non-broadcastable Tensors! {} and {}",
                       op_name, shape_to_string(lhs.dims(), lhs.ndims()),
                       shape_to_string(rhs.dims(), rhs.ndims())));

    out_shape[dim] = std::max(lhs_size, rhs_size);
  }

  return out_shape;
}

using ContiguousBinaryLaunch = void (*)(float *, float *, float *, size_t);
using StridedBinaryLaunch = void (*)(void *, void *, void *, const StrideInfo &,
                                     size_t);
using CpuBinaryOp = float (*)(float, float);

float add_values(float lhs, float rhs) { return lhs + rhs; }
float sub_values(float lhs, float rhs) { return lhs - rhs; }
float mul_values(float lhs, float rhs) { return lhs * rhs; }
float div_values(float lhs, float rhs) { return lhs / rhs; }

void compute_binary_offsets(size_t idx, const TensorShape &shape,
                            const TensorShape &lhs_strides,
                            const TensorShape &rhs_strides, int64_t rank,
                            int64_t &lhs_offset, int64_t &rhs_offset) {
  lhs_offset = 0;
  rhs_offset = 0;

  int64_t remaining = static_cast<int64_t>(idx);
  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    const int64_t coord = remaining % shape[dim];
    remaining /= shape[dim];
    lhs_offset += coord * lhs_strides[dim];
    rhs_offset += coord * rhs_strides[dim];
  }
}

void copy_shape_to_kernel_array(const TensorShape &shape, int64_t *out,
                                int64_t rank) {
  for (int64_t dim = 0; dim < rank; ++dim) {
    out[dim] = shape[dim];
  }
}

Tensor binary_tensor_op(const Tensor &lhs, const Tensor &rhs,
                        const char *op_name,
                        CpuBinaryOp cpu_op,
                        ContiguousBinaryLaunch launch_contiguous,
                        StridedBinaryLaunch launch_strided) {
  ASSERT(lhs.device() == rhs.device(),
         fmt::format("Device mismatch! {} and {}", get_device_name(lhs.device()),
                     get_device_name(rhs.device())));
  ASSERT(lhs.dtype() == rhs.dtype(),
         fmt::format("DType mismatch! {} and {}", get_name(lhs.dtype()),
                     get_name(rhs.dtype())));

  const int64_t out_rank = std::max(lhs.ndims(), rhs.ndims());
  TensorShape out_shape = broadcast_shape(lhs, rhs, op_name);

  Tensor lhs_view = make_broadcast_view(lhs, out_shape, out_rank);
  Tensor rhs_view = make_broadcast_view(rhs, out_shape, out_rank);

  Tensor out = empty(out_shape.data(), out_rank, lhs.dtype(), lhs.device(),
                     lhs.requires_grad() || rhs.requires_grad());

  if (lhs.device() == Device::CPU) {
    const auto *lhs_data = static_cast<const float *>(lhs_view.data());
    const auto *rhs_data = static_cast<const float *>(rhs_view.data());
    auto *out_data = static_cast<float *>(out.data());

    for (size_t idx = 0; idx < out.numel(); ++idx) {
      int64_t lhs_offset = 0;
      int64_t rhs_offset = 0;
      compute_binary_offsets(idx, out_shape, lhs_view.strides(),
                             rhs_view.strides(), out_rank, lhs_offset,
                             rhs_offset);
      out_data[idx] = cpu_op(lhs_data[lhs_offset], rhs_data[rhs_offset]);
    }
    return out;
  }

  if (is_dense_contiguous(lhs_view) && is_dense_contiguous(rhs_view)) {
    launch_contiguous(static_cast<float *>(out.data()),
                      static_cast<float *>(lhs_view.data()),
                      static_cast<float *>(rhs_view.data()), out.numel());
    return out;
  }

  StrideInfo stride_info{};
  stride_info.rank = out_rank;
  for (int64_t dim = 0; dim < out_rank; ++dim) {
    stride_info.output_size[dim] = out_shape[dim];
    stride_info.a_stride[dim] = lhs_view.strides()[dim];
    stride_info.b_stride[dim] = rhs_view.strides()[dim];
  }

  launch_strided(out.data(), lhs_view.data(), rhs_view.data(), stride_info,
                 out.numel());
  return out;
}

void append_tensor_values(fmt::memory_buffer &out, const float *data,
                          const TensorShape &sizes,
                          const TensorShape &strides, int64_t rank,
                          int64_t dim, int64_t offset) {
  if (rank == 0) {
    fmt::format_to(std::back_inserter(out), "{:.4f}", data[offset]);
    return;
  }

  fmt::format_to(std::back_inserter(out), "[");
  for (int64_t i = 0; i < sizes[dim]; ++i) {
    const int64_t next_offset = offset + i * strides[dim];
    if (dim == rank - 1) {
      fmt::format_to(std::back_inserter(out), "{:.4f}", data[next_offset]);
    } else {
      append_tensor_values(out, data, sizes, strides, rank, dim + 1,
                           next_offset);
    }

    if (i != sizes[dim] - 1) {
      fmt::format_to(std::back_inserter(out), ", ");
    }
  }
  fmt::format_to(std::back_inserter(out), "]");
}

} // namespace

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

Tensor full_like(const Tensor &t, float value, bool requires_grad) {
  Tensor out = empty(t.dims().data(), t.ndims(), t.dtype(), t.device(),
                     requires_grad);
  if (t.device() == Device::CUDA) {
    launch_fill(static_cast<float *>(out.data()), out.numel(), value);
  } else {
    std::fill_n(static_cast<float *>(out.data()), out.numel(), value);
  }

  return out;
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
  ASSERT(rank <= static_cast<int64_t>(kMaxTensorDims),
         fmt::format("Tensor rank {} exceeds max rank {}", rank,
                     kMaxTensorDims));

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
Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

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

int64_t Tensor::size(int64_t d) const noexcept { return impl()->sizes[d]; }

int64_t Tensor::ndims() const noexcept { return impl()->ndim; }

Device Tensor::device() const noexcept { return impl()->storage->device; }

DataType Tensor::dtype() const noexcept { return impl()->dtype; }

void *Tensor::data() const noexcept {
  return static_cast<char *>(impl_->storage->ptr);
}

size_t Tensor::numel() const noexcept { return impl_->elems; }

const TensorShape &Tensor::dims() const noexcept { return impl_->sizes; }

const TensorShape &Tensor::strides() const noexcept { return impl_->strides; }

void Tensor::print() const {
  if (!initialized()) {
    fmt::print("Uninitialized Tensor\n");
  } else {
    auto &t = *impl();
    fmt::print(
        "Tensor: [Refcount: {} addr: {} Rank: {} dim({}) "
        "strides({}) "
        "dtype:{} requires_grad:{}]\n\t Storage [Refcount: {} addr: {}]\n",
        impl_.use_count(), fmt::ptr(impl_.get()), t.ndim,
        fmt::join(t.sizes.begin(), t.sizes.begin() + t.ndim, ", "),
        fmt::join(t.strides.begin(), t.strides.begin() + t.ndim, ", "),
        get_name(t.dtype),
        requires_grad(), t.storage.use_count(), t.storage->ptr);
  }
}

void Tensor::print_elms() const { fmt::print("{}", to_string()); }

std::string Tensor::to_string() const {

  if (!initialized()) {
    return "[]";
  }

  // Could be expensive
  auto t = cpu();
  const float *raw_data = static_cast<const float *>(t.data());

  const auto &sizes = dims();
  const auto &stride = strides();

  fmt::memory_buffer out;

  fmt::format_to(std::back_inserter(out), "Tensor: (");
  append_tensor_values(out, raw_data, sizes, stride, ndims(), 0, 0);
  fmt::format_to(std::back_inserter(out), ")\n");

  return fmt::to_string(out);
}

size_t Tensor::total_bytes() const noexcept {
  return element_size(dtype()) * numel();
}

Tensor Tensor::neg() const { return full_like(*this, 0.0f).sub(*this); }

Tensor Tensor::sum(int64_t dim, bool keep_dim) const {
  return ::smollnet::sum(*this, dim, keep_dim);
}

Tensor Tensor::add(float scalar) const { return add(full_like(*this, scalar)); }

Tensor Tensor::mul(const Tensor &other) const {
  Tensor out =
      binary_tensor_op(*this, other, "multiply", mul_values, launch_mul,
                       launch_mul_strided);

  SetupAutograd<MulFunction>(*this, other, out);
  return out;
}

Tensor Tensor::mul(float scalar) const { return mul(full_like(*this, scalar)); }

Tensor Tensor::matmul(const Tensor &other) const {
  return ::smollnet::matmul(*this, other);
}

Tensor Tensor::add(const Tensor &other) const {
  Tensor out =
      binary_tensor_op(*this, other, "add", add_values, launch_add,
                       launch_add_strided);

  SetupAutograd<AddFunction>(*this, other, out);
  return out;
}

Tensor Tensor::sub(const Tensor &other) const {
  Tensor out =
      binary_tensor_op(*this, other, "subtract", sub_values, launch_sub,
                       launch_sub_strided);

  SetupAutograd<SubFunction>(*this, other, out);
  return out;
}

Tensor Tensor::sub(float scalar) const { return sub(full_like(*this, scalar)); }

Tensor Tensor::rsub(float scalar) const {
  return full_like(*this, scalar).sub(*this);
}

Tensor Tensor::div(const Tensor &other) const {
  Tensor out =
      binary_tensor_op(*this, other, "divide", div_values, launch_div,
                       launch_div_strided);

  SetupAutograd<DivFunction>(*this, other, out);
  return out;
}

Tensor Tensor::div(float scalar) const { return div(full_like(*this, scalar)); }

Tensor Tensor::rdiv(float scalar) const {
  return full_like(*this, scalar).div(*this);
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
  return_tensor.impl_ = std::move(view);

  return return_tensor;
}

Tensor Tensor::expand(const TensorShape &new_sz) const {
  const int64_t new_rank = infer_rank(new_sz);
  ASSERT(new_rank > 0 || ndims() == 0,
         "expand requires at least one non-zero dimension");
  return make_broadcast_view(*this, new_sz, new_rank);
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
  ASSERT(l.ndims() == r.ndims(),
         fmt::format("Matmul rank mismatch: {} vs {}", l.ndims(), r.ndims()));
  ASSERT(l.ndims() == 2,
         fmt::format("Matmul currently supports 2D tensors, got rank {}",
                     l.ndims()));

  ASSERT(l.dims()[1] == r.dims()[0],
         fmt::format("Incorrect matrix size! lhs number of rows ({}) not "
                     "equal to rhs number of cols ({})",
                     l.dims()[1], r.dims()[0]));

  ASSERT(l.device() == r.device(),
         fmt::format("Device mismatch! {} and {}", get_device_name(l.device()),
                     get_device_name(r.device())));

  bool needs_grad = any_requires_grad({l, r});
  Tensor new_tensor =
      empty({l.dims()[0], r.dims()[1]}, l.dtype(), l.device(), needs_grad);

  StrideInfo stride_info{};
  stride_info.output_size[0] = new_tensor.size(0);
  stride_info.output_size[1] = new_tensor.size(1);

  const auto &l_strides = l.strides();
  stride_info.a_stride[0] = l_strides[0];
  stride_info.a_stride[1] = l_strides[1];

  const auto &r_strides = r.strides();
  stride_info.b_stride[0] = r_strides[0];
  stride_info.b_stride[1] = r_strides[1];

  stride_info.rank = new_tensor.ndims();

  SizeInfo size_info{};
  size_info.a_size[0] = l.size(0);
  size_info.a_size[1] = l.size(1);

  size_info.b_size[0] = r.size(0);
  size_info.b_size[1] = r.size(1);

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

  ASSERT(dim >= 0,
         fmt::format(
             "Tensor sum(tensor,dim,keep_dim): invalid dim={} t.ndims()={}",
             dim, t.ndims()));

  // build output shape
  TensorShape out_dims{};
  for (int64_t i = 0, o = 0; i < t.ndims(); ++i) {
    if (i != dim) {
      out_dims[o++] = dims[i];
    } else if (keep_dim) {
      out_dims[o++] = 1;
    }
  }

  Tensor new_tensor =
      zeros(out_dims.data(), new_rank, data_type, device, t.requires_grad());

  auto *srcp = t.data();
  auto *dst = new_tensor.data();

  StrideAndSize s_input{};
  s_input.rank = t.ndims();
  copy_shape_to_kernel_array(dims, s_input.size, s_input.rank);
  copy_shape_to_kernel_array(t.strides(), s_input.stride, s_input.rank);

  StrideAndSize s_output{};
  s_output.rank = new_tensor.ndims();
  copy_shape_to_kernel_array(new_tensor.dims(), s_output.size, s_output.rank);
  copy_shape_to_kernel_array(new_tensor.strides(), s_output.stride,
                             s_output.rank);

  launch_sum_dim(dst, srcp, s_input, s_output, dim);

  return new_tensor;
}

Tensor neg(const Tensor &t) { return t.neg(); }

Tensor add(const Tensor &left, const Tensor &right) { return left.add(right); }

Tensor sub(const Tensor &left, const Tensor &right) { return left.sub(right); }

Tensor mul(const Tensor &left, const Tensor &right) { return left.mul(right); }

Tensor div(const Tensor &left, const Tensor &right) { return left.div(right); }

Tensor mse(const Tensor &pred, const Tensor &target) {
  ASSERT(pred.dims() == target.dims(), "");

  bool requires_grad = any_requires_grad({pred, target});
  auto new_tensor = zeros({1}, pred.dtype(), pred.device(), requires_grad);
  launch_mse(new_tensor.data(), pred.data(), target.data(), pred.numel());

  SetupAutograd<MseFunction>(pred, target, new_tensor);
  return new_tensor;
}

Tensor operator+(const Tensor &l, const Tensor &r) { return l.add(r); }

Tensor operator-(const Tensor &t) { return t.neg(); }

Tensor operator-(const Tensor &l, const Tensor &r) { return l.sub(r); }

Tensor operator*(const Tensor &l, const Tensor &r) { return l.mul(r); }

Tensor operator/(const Tensor &l, const Tensor &r) { return l.div(r); }

Tensor operator+(const Tensor &l, float scalar) { return l.add(scalar); }

Tensor operator+(float scalar, const Tensor &r) { return r.add(scalar); }

Tensor operator-(const Tensor &l, float scalar) { return l.sub(scalar); }

Tensor operator-(float scalar, const Tensor &r) { return r.rsub(scalar); }

Tensor operator*(const Tensor &l, float scalar) { return l.mul(scalar); }

Tensor operator*(float scalar, const Tensor &r) { return r.mul(scalar); }

Tensor operator/(const Tensor &l, float scalar) { return l.div(scalar); }

Tensor operator/(float scalar, const Tensor &r) { return r.rdiv(scalar); }

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

Tensor &operator/=(Tensor &l, const Tensor &r) {
  l = l / r;
  return l;
}

Tensor &operator+=(Tensor &l, float scalar) {
  l = l + scalar;
  return l;
}

Tensor &operator-=(Tensor &l, float scalar) {
  l = l - scalar;
  return l;
}

Tensor &operator*=(Tensor &l, float scalar) {
  l = l * scalar;
  return l;
}

Tensor &operator/=(Tensor &l, float scalar) {
  l = l / scalar;
  return l;
}

Tensor empty(const int64_t *dims, size_t rank, DataType t, Device d,
             bool requires_grad) {
  ASSERT(rank <= kMaxTensorDims,
         fmt::format("Tensor rank {} exceeds max rank {}", rank,
                     kMaxTensorDims));

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
  impl->storage = std::move(storage);
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

  if (d == Device::CUDA) {
    launch_fill(static_cast<float *>(tensor.data()), tensor.numel(), 1.0f);
  } else {
    std::fill_n(static_cast<float *>(tensor.data()), tensor.numel(), 1.0f);
  }

  return Tensor{tensor};
}

void manual_seed(unsigned long long seed) {
  cpu_random_generator().seed(static_cast<std::mt19937::result_type>(seed));
  launch_random_init(seed);
}

Tensor rand(const int64_t *dims, size_t rank, DataType t, Device d,
            bool requires_grad) {
  auto tensor = empty(dims, rank, t, d, requires_grad);

  if (d == Device::CUDA) {
    launch_random_fill(tensor.data(), tensor.numel());
  } else {
    auto *data = static_cast<float *>(tensor.data());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto &generator = cpu_random_generator();
    for (size_t i = 0; i < tensor.numel(); ++i) {
      data[i] = dist(generator);
    }
  }

  return Tensor{tensor};
}

Tensor zeros_like(const Tensor &t, bool requires_grad) {
  return zeros(t.dims().data(), t.ndims(), t.dtype(), t.device(),
               requires_grad);
}

Tensor ones_like(const Tensor &t, bool requires_grad) {
  return full_like(t, 1.0f, requires_grad);
}

Tensor rand_like(const Tensor &t, bool requires_grad) {
  return rand(t.dims().data(), t.ndims(), t.dtype(), t.device(), requires_grad);
}

} // namespace smollnet
