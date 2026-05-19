#include "smollnet.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

using namespace pybind11::literals;
namespace py = pybind11;

// Generic templated function to bind tensor creation functions
template <typename FuncType>
void bind_tensor_creation_overloads(pybind11::module &m, const char *func_name,
                                    FuncType &&func) {
  m.def(
      func_name,
      [func](const std::vector<int64_t> &dims,
             smollnet::DataType dtype = smollnet::DataType::f32,
             smollnet::Device device = smollnet::Device::CUDA,
             bool requires_grad = false) {
        if (dims.size() > smollnet::kMaxTensorDims) {
          throw std::invalid_argument("Tensor rank exceeds kMaxTensorDims");
        }
        return func(dims.data(), dims.size(), dtype, device, requires_grad);
      },
      "dims"_a, "dtype"_a = smollnet::DataType::f32,
      "device"_a = smollnet::Device::CUDA, "requires_grad"_a = false);

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
  auto operator()(const int64_t *dims, size_t rank, smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::rand(dims, rank, dtype, device, requires_grad);
  }

  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::rand<N>(dims, dtype, device, requires_grad);
  }
};

struct ZerosFunctor {
  auto operator()(const int64_t *dims, size_t rank, smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::zeros(dims, rank, dtype, device, requires_grad);
  }

  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::zeros<N>(dims, dtype, device, requires_grad);
  }
};

struct OnesFunctor {
  auto operator()(const int64_t *dims, size_t rank, smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::ones(dims, rank, dtype, device, requires_grad);
  }

  template <size_t N>
  auto operator()(const int64_t (&dims)[N], smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::ones<N>(dims, dtype, device, requires_grad);
  }
};

struct EmptyFunctor {
  auto operator()(const int64_t *dims, size_t rank, smollnet::DataType dtype,
                  smollnet::Device device, bool requires_grad) const {
    return smollnet::empty(dims, rank, dtype, device, requires_grad);
  }

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

  m.def("full_like", &smollnet::full_like, "tensor"_a, "value"_a,
        "requires_grad"_a = false);
  m.def("zeros_like", &smollnet::zeros_like, "tensor"_a,
        "requires_grad"_a = false);
  m.def("ones_like", &smollnet::ones_like, "tensor"_a,
        "requires_grad"_a = false);
  m.def("rand_like", &smollnet::rand_like, "tensor"_a,
        "requires_grad"_a = false);
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

      .def("neg", &smollnet::Tensor::neg)
      .def("add", py::overload_cast<const smollnet::Tensor &>(
                      &smollnet::Tensor::add, py::const_))
      .def("add",
           py::overload_cast<float>(&smollnet::Tensor::add, py::const_))
      .def("sub", py::overload_cast<const smollnet::Tensor &>(
                      &smollnet::Tensor::sub, py::const_))
      .def("sub",
           py::overload_cast<float>(&smollnet::Tensor::sub, py::const_))
      .def("rsub", &smollnet::Tensor::rsub)
      .def("sum", &smollnet::Tensor::sum, py::arg("dim"),
           py::arg("keep_dim") = false)
      .def("mul", py::overload_cast<const smollnet::Tensor &>(
                      &smollnet::Tensor::mul, py::const_))
      .def("mul",
           py::overload_cast<float>(&smollnet::Tensor::mul, py::const_))
      .def("div", py::overload_cast<const smollnet::Tensor &>(
                      &smollnet::Tensor::div, py::const_))
      .def("div",
           py::overload_cast<float>(&smollnet::Tensor::div, py::const_))
      .def("rdiv", &smollnet::Tensor::rdiv)
      .def("matmul", &smollnet::Tensor::matmul)

      .def("transpose", &smollnet::Tensor::transpose)
      .def("expand",
           [](const smollnet::Tensor &tensor, const std::vector<int64_t> &dims) {
             if (dims.size() > smollnet::kMaxTensorDims) {
               throw std::invalid_argument("Tensor rank exceeds kMaxTensorDims");
             }
             smollnet::TensorShape shape{};
             std::copy(dims.begin(), dims.end(), shape.begin());
             return tensor.expand(shape);
           })

      .def("cuda", &smollnet::Tensor::cuda)
      .def("cpu", &smollnet::Tensor::cpu)
      .def("copy", &smollnet::Tensor::copy)

      .def(-py::self)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self + float())
      .def(float() + py::self)
      .def(py::self - float())
      .def(float() - py::self)
      .def(py::self * float())
      .def(float() * py::self)
      .def(py::self / float())
      .def(float() / py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= py::self)
      .def(py::self /= py::self)
      .def(py::self += float())
      .def(py::self -= float())
      .def(py::self *= float())
      .def(py::self /= float())
      .def("__matmul__",
           [](const smollnet::Tensor &l, const smollnet::Tensor &r) {
             return l.matmul(r);
           })
      .def("__rmatmul__",
           [](const smollnet::Tensor &r, const smollnet::Tensor &l) {
             return l.matmul(r);
           });

  py::enum_<smollnet::DataType>(m, "DataType")
      .value("f16", smollnet::DataType::f16)
      .value("f32", smollnet::DataType::f32)
      .export_values();

  py::enum_<smollnet::Device>(m, "Device")
      .value("CPU", smollnet::Device::CPU)
      .value("CUDA", smollnet::Device::CUDA)
      .export_values();

  bind_tensor_creation_functions(m);
  m.def("manual_seed", &smollnet::manual_seed, "seed"_a,
        "Initialize the default random generator seed");

  m.def("relu", &smollnet::relu);
  m.def("gelu", &smollnet::gelu);
  m.def("tanh", &smollnet::tanh);
  m.def("sigmoid", &smollnet::sigmoid);

  m.def("matmul", &smollnet::matmul);
  m.def("neg", &smollnet::neg);
  m.def("add", &smollnet::add);
  m.def("sub", &smollnet::sub);
  m.def("mul", &smollnet::mul);
  m.def("div", &smollnet::div);
  m.def("sum", &smollnet::sum, py::arg("tensor"), py::arg("dim"),
        py::arg("keep_dim") = false);

  m.def("mse", &smollnet::mse);

  py::class_<smollnet::SGD>(m, "sgd")
      .def(py::init<std::vector<smollnet::Tensor>, float>(), "params"_a, "lr"_a,
           "Initialize SGD optimizer with parameters and learning rate")

      .def("step", &smollnet::SGD::step, "Perform one optimization step")

      .def("zero_grad", &smollnet::SGD::zero_grad,
           "Zero out gradients of all parameters");

  py::class_<smollnet::Module, std::unique_ptr<smollnet::Module, py::nodelete>>(
      m, "Module")
      .def("forward", &smollnet::Module::forward, py::return_value_policy::move)
      .def("print", &smollnet::Module::print)
      .def("parameters", &smollnet::Module::parameters);

  py::class_<smollnet::LayerNorm>(m, "layer_norm")
      .def(py::init<>(), "Initialize LayerNorm layer")

      // Methods
      .def("compute", &smollnet::LayerNorm::compute, "t"_a,
           "Compute layer normalization")

      .def("__call__", &smollnet::LayerNorm::operator(), "t"_a,
           "Call operator - same as compute")

      .def("forward", &smollnet::LayerNorm::forward, "t"_a, "Forward pass")

      .def("print", &smollnet::LayerNorm::print, "Print layer information")

      .def("parameters", &smollnet::LayerNorm::parameters,
           "Get all parameters of the layer")

      // Public member access
      .def_readwrite("weights", &smollnet::LayerNorm::weights,
                     "Layer normalization weights")
      .def_readwrite("bias", &smollnet::LayerNorm::bias,
                     "Layer normalization bias");

  py::class_<smollnet::Linear, smollnet::Module>(m, "Linear")
      .def(py::init<int64_t, int64_t>(), "in_dim"_a, "out_dim"_a,
           "Initialize Linear layer with input and output dimensions")

      .def("forward", &smollnet::Linear::forward, "t"_a, "Forward pass")
      .def("parameters", &smollnet::Linear::parameters, "Get all parameters")
      .def("print", &smollnet::Linear::print, "Print layer information")

      .def_readwrite("weights", &smollnet::Linear::weights)
      .def_readwrite("bias", &smollnet::Linear::bias);

  // ReLU activation
  py::class_<smollnet::ReLU, smollnet::Module>(m, "ReLU")
      .def(py::init<>(), "Initialize ReLU activation")

      .def("forward", &smollnet::ReLU::forward, "t"_a, "Forward pass")
      .def("parameters", &smollnet::ReLU::parameters, "Get all parameters")
      .def("print", &smollnet::ReLU::print, "Print activation information");

  // GeLU activation
  py::class_<smollnet::GeLU, smollnet::Module>(m, "GeLU")
      .def(py::init<>(), "Initialize GeLU activation")

      .def("forward", &smollnet::GeLU::forward, "t"_a, "Forward pass")
      .def("parameters", &smollnet::GeLU::parameters, "Get all parameters")
      .def("print", &smollnet::GeLU::print, "Print activation information");

  // Dense network - this is more complex due to the variadic template
  // constructor
  py::class_<smollnet::Dense>(m, "Dense")
      .def(py::init([](const py::args& args) {
        std::vector<std::unique_ptr<smollnet::Module>> modules;
        for (auto &arg : args) {
          if (py::isinstance<smollnet::Linear>(arg)) {
            modules.push_back(std::make_unique<smollnet::Linear>(
                arg.cast<smollnet::Linear>()));
          } else if (py::isinstance<smollnet::ReLU>(arg)) {
            modules.push_back(
                std::make_unique<smollnet::ReLU>(arg.cast<smollnet::ReLU>()));
          } else if (py::isinstance<smollnet::GeLU>(arg)) {
            modules.push_back(
                std::make_unique<smollnet::GeLU>(arg.cast<smollnet::GeLU>()));
          } else if (py::isinstance<smollnet::LayerNorm>(arg)) {
            modules.push_back(std::make_unique<smollnet::LayerNorm>(
                arg.cast<smollnet::LayerNorm>()));
          }
        }
        return smollnet::Dense(std::move(modules));
      }))

      .def("forward", &smollnet::Dense::forward, "input"_a,
           "Forward pass through the network")
      .def("parameters", &smollnet::Dense::parameters,
           "Get all parameters from all layers")
      .def("train", &smollnet::Dense::train, "input"_a, "targets"_a,
           "lr"_a = 0.0001f, "num_epochs"_a = 32, "Train the network")
      .def("print", &smollnet::Dense::print, "Print network information")
      .def("print_grads", &smollnet::Dense::print_grads, "Print gradients");
}
