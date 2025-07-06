#include "tensor.hpp"

#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(tensor, m) {
    py::class_<smollnet::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def_property_readonly("initialized", &smollnet::Tensor::initialized)
        .def("ndims", &smollnet::Tensor::ndims)
        .def("size", &smollnet::Tensor::size)
        .def("add", &smollnet::Tensor::add);
}
