#pragma once
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace smollnet {
template <typename Derived>
struct Module {
    Tensor forward(Tensor& t) {
        return static_cast<Derived*>(this)->forward_impl(t);
    }
};

struct Linear : Module<Linear> {
    Tensor weights, bias;
    Linear(int64_t in_dim, int64_t out_dim) { /* Initialize weights, bias */ }
    Tensor forward_impl(Tensor& t) { /* matmul + bias */ }
};

struct ReLU : Module<ReLU> {
    Tensor forward_impl(Tensor& t) { /* Apply ReLU */ }
};

template <typename... Modules>
class Dense {
public:
    Dense(Modules&&... modules) : layers_(std::forward<Modules>(modules)...) {}

    Tensor forward(const Tensor& input) const {
        Tensor output = input;
        std::apply([&](const auto&... layer) {
            ((output = layer.forward(output)), ...);
        }, layers_);
        return output;
    }

private:
    std::tuple<Modules...> layers_;
    Device device_ = Device::CPU;
    DataType dtype_ = DataType::f32;
};

} // namespace smollnet
