#pragma once

#include "tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace pybind11::literals;
namespace py = pybind11;

// Generic templated function to bind tensor creation functions
template <typename FuncType>
void bind_tensor_creation_overloads(pybind11::module &m, const char *func_name,
                                    FuncType &&func) {
  // 1D version
  m.def(
      func_name,
      [func](int64_t dim0, smollnet::DataType dtype = smollnet::DataType::f32,
             smollnet::Device device = smollnet::Device::CUDA,
             bool requires_grad = false) {
        int64_t dims_array[1] = {dim0};
        return func.template operator()<1>(dims_array, dtype, device,
                                           requires_grad);
      },
      "dim0"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 2D version
  m.def(
      func_name,
      [func](int64_t dim0, int64_t dim1,
             smollnet::DataType dtype = smollnet::DataType::f32,
             smollnet::Device device = smollnet::Device::CUDA,
             bool requires_grad = false) {
        int64_t dims_array[2] = {dim0, dim1};
        return func.template operator()<2>(dims_array, dtype, device,
                                           requires_grad);
      },
      "dim0"_a, "dim1"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 3D version
  m.def(
      func_name,
      [func](int64_t dim0, int64_t dim1, int64_t dim2,
             smollnet::DataType dtype = smollnet::DataType::f32,
             smollnet::Device device = smollnet::Device::CUDA,
             bool requires_grad = false) {
        int64_t dims_array[3] = {dim0, dim1, dim2};
        return func.template operator()<3>(dims_array, dtype, device,
                                           requires_grad);
      },
      "dim0"_a, "dim1"_a, "dim2"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);
}

// Function objects for each tensor creation function
struct RandFunctor {
  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::rand<N>(dims, dtype, device, requires_grad);
  }
};

struct ZerosFunctor {
  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::zeros<N>(dims, dtype, device, requires_grad);
  }
};

struct OnesFunctor {
  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::ones<N>(dims, dtype, device, requires_grad);
  }
};

struct EmptyFunctor {
  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::empty<N>(dims, dtype, device, requires_grad);
  }
};

void bind_tensor_creation_functions(pybind11::module &m) {
  bind_tensor_creation_overloads(m, "rand", RandFunctor{});
  bind_tensor_creation_overloads(m, "zeros", ZerosFunctor{});
  bind_tensor_creation_overloads(m, "ones", OnesFunctor{});
  bind_tensor_creation_overloads(m, "empty", EmptyFunctor{});
}

// Single module definition
PYBIND11_MODULE(smollnet, m) {
  m.doc() = "SmollNet Tensor Module";

  // Bind the Tensor class
  py::class_<smollnet::Tensor>(m, "Tensor")
      .def("__repr__", &smollnet::Tensor::to_string,
           "String representation of the tensor")

      .def("initialized", &smollnet::Tensor::initialized)

      .def("size", &smollnet::Tensor::size)
      .def("ndims", &smollnet::Tensor::ndims)
      .def("numel", &smollnet::Tensor::numel)
      .def("dims", &smollnet::Tensor::dims)
      .def("strides", &smollnet::Tensor::strides)

      .def("device", &smollnet::Tensor::device)
      .def("dtype", &smollnet::Tensor::dtype)

      .def("backward", &smollnet::Tensor::backward,
           py::arg("grad_output") = smollnet::Tensor{})
      .def("zero_grad", &smollnet::Tensor::zero_grad)
      .def("requires_grad", &smollnet::Tensor::requires_grad)
      .def("grad", &smollnet::Tensor::grad)

      .def("print", &smollnet::Tensor::print)
      .def("print_elms", &smollnet::Tensor::print_elms)

      .def("add", &smollnet::Tensor::add)
      .def("sub", &smollnet::Tensor::sub)
      .def("sum", &smollnet::Tensor::sum, py::arg("dim"),
           py::arg("keep_dim") = false)
      .def("mul", &smollnet::Tensor::mul)
      .def("matmul", &smollnet::Tensor::matmul)

      .def("transpose", &smollnet::Tensor::transpose)
      .def("expand", &smollnet::Tensor::expand)

      .def("cuda", &smollnet::Tensor::cuda)
      .def("cpu", &smollnet::Tensor::cpu)
      .def("copy", &smollnet::Tensor::copy)

      .def("__add__", &smollnet::operator+)
      .def("__sub__", &smollnet::operator-)
      .def("__mul__", &smollnet::operator*)
      .def("__radd__", &smollnet::operator+)
      .def("__rsub__",
           [](const smollnet::Tensor &rhs, const smollnet::Tensor &lhs) {
             return smollnet::operator-(lhs, rhs);
           })
      .def("__rmul__", &smollnet::operator*)

      // In-place operators (optional)
      .def("__iadd__",
           [](smollnet::Tensor &self,
              const smollnet::Tensor &other) -> smollnet::Tensor & {
             self = self + other;
             return self;
           })
      .def("__isub__",
           [](smollnet::Tensor &self,
              const smollnet::Tensor &other) -> smollnet::Tensor & {
             self = self - other;
             return self;
           })
      .def("__imul__",
           [](smollnet::Tensor &self,
              const smollnet::Tensor &other) -> smollnet::Tensor & {
             self = self * other;
             return self;
           });

  py::enum_<smollnet::DataType>(m, "DataType")
      .value("f32", smollnet::DataType::f32)
      .export_values();

  py::enum_<smollnet::Device>(m, "Device")
      .value("CPU", smollnet::Device::CPU)
      .value("CUDA", smollnet::Device::CUDA)
      .export_values();

  bind_tensor_creation_functions(m);

  m.def("relu", &smollnet::relu);
  m.def("gelu", &smollnet::gelu);
  m.def("tanh", &smollnet::tanh);
  m.def("sigmoid", &smollnet::sigmoid);

  m.def("matmul", &smollnet::matmul);
  m.def("mul", &smollnet::mul);
  m.def("sum", &smollnet::sum, py::arg("tensor"), py::arg("dim"),
        py::arg("keep_dim") = false);

  m.def("mse", &smollnet::mse);
}
