#include <tensor.hpp>
#include <types.hpp>

using namespace smollnet;

int main() {
  // contents: uninitialized float32 on GPU
  Tensor a = empty({2, 3}, DataType::f32, Device::CUDA);
  a.print();

  // contents: all ones, shape [2, 3], stride [3, 1]
  Tensor b = ones({2, 3}, DataType::f32, Device::CUDA);
  b.print();

  // elementwise addition, c[i,j] = a[i,j] + b[i,j]
  Tensor c = a + b;
  c.print();

  // view: shape [3, 2], stride adjusted, no data copied
  Tensor d = c.transpose(0, 1);
  d.print();


//   Tensor e = d.contiguous();
//   // materialized in row-major order; storage copied

//   Tensor f = e.slice(1, 0, 2);
//   // selects rows 0 and 1, shape changes, same storage aliased

//   f.backward();
  // triggers autograd graph execution if requires_grad == true
}
