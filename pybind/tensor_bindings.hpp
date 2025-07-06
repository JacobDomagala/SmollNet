#pragma once

#include "tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace pybind11::literals;
namespace py = pybind11;

void bind_rand_overloads(pybind11::module &m) {
  // 1D version
  m.def(
      "rand",
      [](int64_t dim0, smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[1] = {dim0};
        return smollnet::rand<1>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 2D version
  m.def(
      "rand",
      [](int64_t dim0, int64_t dim1,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[2] = {dim0, dim1};
        return smollnet::rand<2>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 3D version
  m.def(
      "rand",
      [](int64_t dim0, int64_t dim1, int64_t dim2,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[3] = {dim0, dim1, dim2};
        return smollnet::rand<3>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dim2"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);
}

void bind_zeros_overloads(pybind11::module &m) {
  // 1D version
  m.def(
      "zeros",
      [](int64_t dim0, smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[1] = {dim0};
        return smollnet::zeros<1>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 2D version
  m.def(
      "zeros",
      [](int64_t dim0, int64_t dim1,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[2] = {dim0, dim1};
        return smollnet::zeros<2>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 3D version
  m.def(
      "zeros",
      [](int64_t dim0, int64_t dim1, int64_t dim2,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[3] = {dim0, dim1, dim2};
        return smollnet::zeros<3>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dim2"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);
}

void bind_ones_overloads(pybind11::module &m) {
  // 1D version
  m.def(
      "ones",
      [](int64_t dim0, smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[1] = {dim0};
        return smollnet::ones<1>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 2D version
  m.def(
      "ones",
      [](int64_t dim0, int64_t dim1,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[2] = {dim0, dim1};
        return smollnet::ones<2>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 3D version
  m.def(
      "ones",
      [](int64_t dim0, int64_t dim1, int64_t dim2,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[3] = {dim0, dim1, dim2};
        return smollnet::ones<3>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dim2"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);
}

void bind_empty_overloads(pybind11::module &m) {
  // 1D version
  m.def(
      "empty",
      [](int64_t dim0, smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[1] = {dim0};
        return smollnet::empty<1>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 2D version
  m.def(
      "empty",
      [](int64_t dim0, int64_t dim1,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[2] = {dim0, dim1};
        return smollnet::empty<2>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

  // 3D version
  m.def(
      "empty",
      [](int64_t dim0, int64_t dim1, int64_t dim2,
         smollnet::DataType dtype = smollnet::DataType::f32,
         smollnet::Device device = smollnet::Device::CUDA,
         bool requires_grad = false) {
        int64_t dims_array[3] = {dim0, dim1, dim2};
        return smollnet::empty<3>(dims_array, dtype, device, requires_grad);
      },
      "dim0"_a, "dim1"_a, "dim2"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);
}

// Single module definition
PYBIND11_MODULE(smollnet, m) {
  m.doc() = "SmollNet Tensor Module";

  // Bind the Tensor class
  py::class_<smollnet::Tensor>(m, "Tensor")
      .def(py::init<>())
      .def("initialized", &smollnet::Tensor::initialized)

      .def("size", &smollnet::Tensor::size)
      .def("ndims", &smollnet::Tensor::ndims)
      .def("numel", &smollnet::Tensor::numel)
      .def("dims", &smollnet::Tensor::dims)
      .def("strides", &smollnet::Tensor::strides)

      .def("device", &smollnet::Tensor::device)
      .def("dtype", &smollnet::Tensor::dtype)

      .def("backward", &smollnet::Tensor::backward)
      .def("zero_grad", &smollnet::Tensor::zero_grad)
      .def("requires_grad", &smollnet::Tensor::requires_grad)
      .def("grad", &smollnet::Tensor::grad)

      .def("print", &smollnet::Tensor::print)
      .def("print_elms", &smollnet::Tensor::print_elms)

      .def("add", &smollnet::Tensor::add)
      .def("sub", &smollnet::Tensor::sub)
      .def("sum", &smollnet::Tensor::sum)
      .def("mul", &smollnet::Tensor::mul)
      .def("matmul", &smollnet::Tensor::matmul)

      .def("transpose", &smollnet::Tensor::transpose)
      .def("expand", &smollnet::Tensor::expand)

      .def("cuda", &smollnet::Tensor::cuda)
      .def("cpu", &smollnet::Tensor::cpu)
      .def("copy", &smollnet::Tensor::copy);

  py::enum_<smollnet::DataType>(m, "DataType")
      .value("f32", smollnet::DataType::f32)
      .export_values();

  py::enum_<smollnet::Device>(m, "Device")
      .value("CPU", smollnet::Device::CPU)
      .value("CUDA", smollnet::Device::CUDA)
      .export_values();

  bind_rand_overloads(m);
  bind_zeros_overloads(m);
  bind_ones_overloads(m);
  bind_empty_overloads(m);
}
