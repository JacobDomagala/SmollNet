#include "autograd.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include <queue>
#include <unordered_set>

namespace smollnet {

// Helper function to check if any tensor requires grad
bool any_requires_grad(const std::vector<Tensor> &tensors) {
  for (const auto &tensor : tensors) {
    if (tensor.impl() && tensor.impl()->requires_grad) {
      return true;
    }
  }
  return false;
}

// Create a gradient tensor with the same shape as input
Tensor create_grad_tensor(const Tensor &tensor) {
  return zeros(tensor.dims().data(), tensor.ndims(), tensor.dtype(),
               tensor.device());
}

// AddFunction implementation
AddFunction::AddFunction(const Tensor &lhs, const Tensor &rhs) {
  inputs = {lhs, rhs};
  needs_input_grad = {lhs.impl() && lhs.requires_grad(),
                      rhs.impl() && rhs.requires_grad()};
}

std::vector<Tensor>
AddFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "AddFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(2);

  if (needs_input_grad[0]) {
    // Gradient w.r.t. lhs is just the incoming gradient
    grad_inputs[0] = grad_outputs[0];
  }

  if (needs_input_grad[1]) {
    // Gradient w.r.t. rhs is just the incoming gradient
    grad_inputs[1] = grad_outputs[0];
  }

  return grad_inputs;
}

// SubFunction implementation
SubFunction::SubFunction(const Tensor &lhs, const Tensor &rhs) {
  inputs = {lhs, rhs};
  needs_input_grad = {lhs.impl() && lhs.requires_grad(),
                      rhs.impl() && rhs.requires_grad()};
}

std::vector<Tensor>
SubFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "SubFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(2);

  if (needs_input_grad[0]) {
    // Gradient w.r.t. lhs is the incoming gradient
    grad_inputs[0] = grad_outputs[0];
  }

  if (needs_input_grad[1]) {
    // Gradient w.r.t. rhs is negative of incoming gradient
    auto neg_grad =
        zeros(grad_outputs[0].dims().data(), grad_outputs[0].ndims(),
              grad_outputs[0].dtype(), grad_outputs[0].device());

    // TODO: Implement negate operation in kernels
    grad_inputs[1] = neg_grad.sub(const_cast<Tensor &>(grad_outputs[0]));
  }

  return grad_inputs;
}

// MatmulFunction implementation
MatmulFunction::MatmulFunction(const Tensor &lhs, const Tensor &rhs) {
  inputs = {lhs, rhs};
  lhs_shape = lhs.dims();
  rhs_shape = rhs.dims();
  needs_input_grad = {lhs.impl() && lhs.requires_grad(),
                      rhs.impl() && rhs.requires_grad()};
}

std::vector<Tensor>
MatmulFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "MatmulFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(2);
  auto &grad_output = const_cast<Tensor &>(grad_outputs[0]);

  if (needs_input_grad[0]) {
    // grad_lhs = grad_output @ rhs.T
    auto rhs_t = inputs[1].transpose(0, 1);
    grad_inputs[0] = matmul(grad_output, rhs_t);
  }

  if (needs_input_grad[1]) {
    // grad_rhs = lhs.T @ grad_output
    auto lhs_t = inputs[0].transpose(0, 1);
    grad_inputs[1] = matmul(lhs_t, grad_output);
  }

  return grad_inputs;
}

// ReLUFunction implementation
ReLUFunction::ReLUFunction(const Tensor &input) {
  inputs = {input};
  needs_input_grad = {input.impl() && input.requires_grad()};
}

std::vector<Tensor>
ReLUFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "ReLUFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(1);

  if (needs_input_grad[0]) {
    auto grad_input = create_grad_tensor(inputs[0]);
    launch_relu_grad(grad_input.data(), grad_outputs.front().data(),
                     grad_input.numel());

    grad_inputs[0] = grad_input;
  }

  return grad_inputs;
}

// TanhFunction implementation
TanhFunction::TanhFunction(const Tensor &input) {
  inputs = {input};
  needs_input_grad = {input.impl() && input.requires_grad()};
}

std::vector<Tensor>
TanhFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "TanhFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(1);

  if (needs_input_grad[0]) {
    auto grad_input = create_grad_tensor(inputs[0]);
    launch_tanh_grad(grad_input.data(), grad_outputs.front().data(),
                     grad_input.numel());
    grad_inputs[0] = grad_input;
  }

  return grad_inputs;
}

// SigmoidFunction implementation
SigmoidFunction::SigmoidFunction(const Tensor &input) {
  inputs = {input};
  needs_input_grad = {input.impl() && input.requires_grad()};
}

std::vector<Tensor>
SigmoidFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "SigmoidFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(1);

  if (needs_input_grad[0]) {

    auto grad_input = create_grad_tensor(inputs[0]);
    launch_sigmoid_grad(grad_input.data(), grad_outputs.front().data(),
                        grad_input.numel());
    grad_inputs[0] = grad_input;
  }

  return grad_inputs;
}

// SumFunction implementation
SumFunction::SumFunction(const Tensor &input, int64_t dim) : dim_(dim) {
  inputs = {input};
  input_shape_ = input.dims();
  needs_input_grad = {input.impl() && input.requires_grad()};
}

std::vector<Tensor>
SumFunction::backward(const std::vector<Tensor> &grad_outputs) {
  ASSERT(grad_outputs.size() == 1,
         "SumFunction expects exactly one gradient output");

  std::vector<Tensor> grad_inputs(1);

  if (needs_input_grad[0]) {
    // Sum gradient: broadcast the gradient back to original shape
    auto grad_input = create_grad_tensor(inputs[0]);
    // TODO: Implement proper broadcasting for sum backward
    // For now, create a tensor of ones with the original shape
    grad_inputs[0] = ones(input_shape_.data(), inputs[0].ndims(),
                          inputs[0].dtype(), inputs[0].device());
  }

  return grad_inputs;
}

// Main backward function - implements reverse-mode automatic differentiation
void backward(Tensor &tensor, const Tensor &grad_output) {
  ASSERT(tensor.impl(), "Cannot compute gradients for null tensor");
  ASSERT(tensor.requires_grad(), "Tensor does not require gradients");

  auto *meta = tensor.autograd();

  // Initialize gradient if not provided
  Tensor grad;
  if (grad_output.initialized()) {
    grad = grad_output;
  } else {
    grad = ones(tensor.dims().data(), tensor.ndims(), tensor.dtype(),
                tensor.device());
  }

  // Topological sort using DFS for gradient computation
  std::queue<std::pair<Tensor, Tensor>> ready_queue; // (tensor, gradient)
  std::unordered_set<TensorImpl *> visited;

  ready_queue.push({tensor, grad});

  while (!ready_queue.empty()) {
    auto [current_tensor, current_grad] = ready_queue.front();
    ready_queue.pop();

    auto *current_impl = current_tensor.impl();
    if (!current_impl || visited.count(current_impl))
      continue;

    visited.insert(current_impl);

    auto *current_meta = static_cast<AutogradMeta *>(current_impl->grad);
    if (!current_meta)
      continue;

    // Accumulate gradient
    if (!current_meta->grad.initialized()) {
      current_meta->grad = current_grad;
    } else {
      current_meta->grad =
          current_meta->grad.add(const_cast<Tensor &>(current_grad));
    }

    // If this is a leaf node, we're done with this path
    if (current_meta->is_leaf)
      continue;

    // Compute gradients for inputs if we have a gradient function
    if (current_meta->grad_fn) {
      current_meta->grad_fn->print();
      auto input_grads = current_meta->grad_fn->backward({current_grad});

      for (size_t i = 0; i < input_grads.size(); ++i) {
        if (input_grads[i].initialized() && i < current_meta->grad_fn->inputs.size()) {
          ready_queue.push({current_meta->grad_fn->inputs[i], input_grads[i]});
        }
      }
    }
  }
}

} // namespace smollnet
