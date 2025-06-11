#include <smollnet.hpp>

#include <fmt/core.h>

using namespace smollnet;

int main() {
  constexpr int input_size = 3;

  auto norm = LayerNorm();
  Tensor input = rand({input_size, 6}, DataType::f32, Device::CUDA);
  auto out = norm(input);

  input.print();
  input.print_elms();

  out.print();
  out.print_elms();

}
