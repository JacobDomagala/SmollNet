#pragma once

#include "tensor.hpp"

#include <memory>
#include <vector>

namespace smollnet {

class Tensor;
struct TensorImpl;

struct Function {
  virtual ~Function() = default;
  virtual std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) = 0;
  virtual void print() const = 0;

  std::vector<Tensor> inputs;
  std::vector<bool> needs_input_grad;
};

// Autograd metadata for tensors
struct AutogradMeta {
  std::shared_ptr<Function> grad_fn = nullptr;
  Tensor grad;
  bool is_leaf = true;
};

// Specific function implementations
struct AddFunction : Function {
  AddFunction(const Tensor &lhs, const Tensor &rhs);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("AddFunction\n"); }
};

struct SubFunction : Function {
  SubFunction(const Tensor &lhs, const Tensor &rhs);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("SubFunction\n"); }
};


struct MulFunction : Function {
  MulFunction(const Tensor &lhs, const Tensor &rhs);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("MulFunction\n"); }
};

struct MatmulFunction : Function {
  MatmulFunction(const Tensor &lhs, const Tensor &rhs);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("MatmulFunction\n"); }

private:
  std::array<int64_t, 3> lhs_shape;
  std::array<int64_t, 3> rhs_shape;
};

struct ReLUFunction : Function {
  ReLUFunction(const Tensor &input);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("ReLUFunction\n"); }
};

struct GeLUFunction : Function {
  GeLUFunction(const Tensor &input);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("GeLUFunction\n"); }
};

struct TanhFunction : Function {
  TanhFunction(const Tensor &input);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("TanhFunction\n"); }
};

struct SigmoidFunction : Function {
  SigmoidFunction(const Tensor &input);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("SigmoidFunction\n"); }
};

struct SumFunction : Function {
  SumFunction(const Tensor &input, int64_t dim);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("SumFunction\n"); }

private:
  int64_t dim_;
  std::array<int64_t, 3> input_shape_;
};

struct MseFunction : Function {
  MseFunction(const Tensor &pred, const Tensor &tgt);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("MseFunction\n"); }

private:
  size_t N;
  int64_t dim_;
  std::array<int64_t, 3> input_shape_;
};

struct LayerNormFunction : Function {
  LayerNormFunction(const Tensor &mean, const Tensor &variance,
                                     const Tensor &normalized, const Tensor& original,
                                     const Tensor &scale, const Tensor &bias);
  std::vector<Tensor>
  backward(const std::vector<Tensor> &grad_outputs) override;
  void print() const override { printf("LayerNorm\n"); }
};

// Autograd engine functions
void backward(Tensor &tensor, const Tensor &grad_output = Tensor());
bool any_requires_grad(const std::vector<Tensor> &tensors);
Tensor create_grad_tensor(const Tensor &tensor);

} // namespace smollnet
