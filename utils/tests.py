#!/usr/bin/env python3
"""
Comprehensive test script for SmollNet neural network library.
Tests tensor operations, neural network layers, and training functionality.
"""

import smollnet

def test_tensor_creation():
    """Test tensor creation functions."""
    print("=== Testing Tensor Creation ===")

    # Test 1D tensors
    print("Creating 1D tensors...")
    t1d_zeros = smollnet.zeros(5)
    t1d_ones = smollnet.ones(5)
    t1d_rand = smollnet.rand(5)
    t1d_empty = smollnet.empty(5)

    print(f"1D zeros: {t1d_zeros}")
    print(f"1D ones: {t1d_ones}")
    print(f"1D rand: {t1d_rand}")
    print(f"1D empty: {t1d_empty}")

    # Test 2D tensors
    print("\nCreating 2D tensors...")
    t2d_zeros = smollnet.zeros(3, 4)
    t2d_ones = smollnet.ones(3, 4)
    t2d_rand = smollnet.rand(3, 4)

    print(f"2D zeros shape: {t2d_zeros.dims()}")
    print(f"2D ones shape: {t2d_ones.dims()}")
    print(f"2D rand shape: {t2d_rand.dims()}")

    # Test 3D tensors
    print("\nCreating 3D tensors...")
    t3d_zeros = smollnet.zeros(2, 3, 4)
    t3d_ones = smollnet.ones(2, 3, 4)
    t3d_rand = smollnet.rand(2, 3, 4)

    print(f"3D zeros shape: {t3d_zeros.dims()}")
    print(f"3D ones shape: {t3d_ones.dims()}")
    print(f"3D rand shape: {t3d_rand.dims()}")

    # Test with different data types and devices
    print("\nTesting different data types and devices...")
    t_cpu = smollnet.zeros(3, 3, dtype=smollnet.DataType.f32, device=smollnet.Device.CPU)
    t_cuda = smollnet.zeros(3, 3, dtype=smollnet.DataType.f32, device=smollnet.Device.CUDA)

    print(f"CPU tensor device: {t_cpu.device()}")
    print(f"CUDA tensor device: {t_cuda.device()}")

    print("✓ Tensor creation tests passed!\n")

def test_tensor_operations():
    """Test basic tensor operations."""
    print("=== Testing Tensor Operations ===")

    # Create test tensors
    a = smollnet.rand(3, 3)
    b = smollnet.rand(3, 3)

    print("Testing basic arithmetic operations...")

    # Addition
    c_add = a + b
    print(f"Addition result shape: {c_add.dims()}")

    # Subtraction
    c_sub = a - b
    print(f"Subtraction result shape: {c_sub.dims()}")

    # Multiplication
    c_mul = a * b
    print(f"Multiplication result shape: {c_mul.dims()}")

    # Matrix multiplication
    c_matmul = smollnet.matmul(a, b)
    print(f"Matrix multiplication result shape: {c_matmul.dims()}")

    # Test in-place operations
    a_copy = a.copy()
    a_copy += b
    print(f"In-place addition result shape: {a_copy.dims()}")

    # Test tensor methods
    print(f"Tensor a size: {a.size()}")
    print(f"Tensor a ndims: {a.ndims()}")
    print(f"Tensor a numel: {a.numel()}")
    print(f"Tensor a strides: {a.strides()}")

    # Test sum operation
    sum_result = smollnet.sum(a, 0)
    print(f"Sum along dim 0 shape: {sum_result.dims()}")

    # Test transpose
    a_t = a.transpose()
    print(f"Transpose result shape: {a_t.dims()}")

    print("✓ Tensor operations tests passed!\n")

def test_activation_functions():
    """Test activation functions."""
    print("=== Testing Activation Functions ===")

    # Create test tensor
    x = smollnet.rand(3, 4)

    # Test ReLU
    relu_result = smollnet.relu(x)
    print(f"ReLU result shape: {relu_result.dims()}")

    # Test GeLU
    gelu_result = smollnet.gelu(x)
    print(f"GeLU result shape: {gelu_result.dims()}")

    # Test Tanh
    tanh_result = smollnet.tanh(x)
    print(f"Tanh result shape: {tanh_result.dims()}")

    # Test Sigmoid
    sigmoid_result = smollnet.sigmoid(x)
    print(f"Sigmoid result shape: {sigmoid_result.dims()}")

    print("✓ Activation function tests passed!\n")

def test_neural_network_layers():
    """Test neural network layers."""
    print("=== Testing Neural Network Layers ===")

    # Test Linear layer
    print("Testing Linear layer...")
    linear = smollnet.Linear(4, 3)
    input_tensor = smollnet.rand(2, 4)  # batch_size=2, input_dim=4
    output = linear.forward(input_tensor)
    print(f"Linear layer output shape: {output.dims()}")

    # Test ReLU layer
    print("Testing ReLU layer...")
    relu_layer = smollnet.ReLU()
    relu_output = relu_layer.forward(output)
    print(f"ReLU layer output shape: {relu_output.dims()}")

    # Test GeLU layer
    print("Testing GeLU layer...")
    gelu_layer = smollnet.GeLU()
    gelu_output = gelu_layer.forward(output)
    print(f"GeLU layer output shape: {gelu_output.dims()}")

    # Test LayerNorm
    print("Testing LayerNorm layer...")
    layer_norm = smollnet.layer_norm()
    norm_output = layer_norm.forward(output)
    print(f"LayerNorm output shape: {norm_output.dims()}")

    # Test parameter access
    linear_params = linear.parameters()
    print(f"Linear layer has {len(linear_params)} parameters")

    print("✓ Neural network layer tests passed!\n")

def test_dense_network():
    """Test Dense (sequential) network."""
    print("=== Testing Dense Network ===")

    # Create a simple dense network
    dense = smollnet.Dense(
        smollnet.Linear(4, 8),
        smollnet.ReLU(),
        smollnet.Linear(8, 4),
        smollnet.GeLU(),
        smollnet.Linear(4, 2)
    )

    # Test forward pass
    input_tensor = smollnet.rand(3, 4)  # batch_size=3, input_dim=4
    output = dense.forward(input_tensor)
    print(f"Dense network output shape: {output.dims()}")

    # Test parameter access
    all_params = dense.parameters()
    print(f"Dense network has {len(all_params)} parameters")

    # Print network structure
    dense.print()

    print("✓ Dense network tests passed!\n")

def test_optimizer():
    """Test SGD optimizer."""
    print("=== Testing SGD Optimizer ===")

    # Create a simple network
    linear = smollnet.Linear(3, 2)

    # Get parameters
    params = linear.parameters()
    print(f"Network has {len(params)} parameters")

    # Create optimizer
    optimizer = smollnet.sgd(params, lr=0.01)

    # Create dummy data
    x = smollnet.rand(4, 3, requires_grad=False)
    y = smollnet.rand(4, 2, requires_grad=False)

    # Forward pass
    output = linear.forward(x)

    # Compute loss (MSE)
    loss = smollnet.mse(output, y)
    print(f"Initial loss: {loss}")

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    print("✓ SGD optimizer tests passed!\n")

def test_training_loop():
    """Test a simple training loop."""
    print("=== Testing Training Loop ===")

    # Create a simple regression problem
    input_dim = 4
    output_dim = 2
    batch_size = 8

    # Create network
    network = smollnet.Dense(
        smollnet.Linear(input_dim, 8),
        smollnet.ReLU(),
        smollnet.Linear(8, output_dim)
    )

    # Create optimizer
    params = network.parameters()
    optimizer = smollnet.sgd(params, lr=0.01)

    print(f"Training network with {len(params)} parameters")

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        # Generate random data
        x = smollnet.rand(batch_size, input_dim)
        y = smollnet.rand(batch_size, output_dim)

        # Forward pass
        output = network.forward(x)

        # Compute loss
        loss = smollnet.mse(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")

    print("✓ Training loop tests passed!\n")

def test_gradient_computation():
    """Test gradient computation and backpropagation."""
    print("=== Testing Gradient Computation ===")

    # Create tensors with gradient tracking
    x = smollnet.rand(2, 3, requires_grad=True)
    y = smollnet.rand(2, 3, requires_grad=True)

    print(f"x requires_grad: {x.requires_grad()}")
    print(f"y requires_grad: {y.requires_grad()}")

    # Compute some operations
    z = x + y
    loss = smollnet.sum(z, 0)

    # Backward pass
    loss.backward()

    # Check gradients
    print(f"x has gradient: {x.grad().initialized()}")
    print(f"y has gradient: {y.grad().initialized()}")

    # Zero gradients
    x.zero_grad()
    y.zero_grad()

    print("✓ Gradient computation tests passed!\n")

def test_device_operations():
    """Test device operations (CPU/CUDA)."""
    print("=== Testing Device Operations ===")

    # Create tensors on different devices
    cpu_tensor = smollnet.zeros(3, 3, device=smollnet.Device.CPU)
    cuda_tensor = smollnet.zeros(3, 3, device=smollnet.Device.CUDA)

    print(f"CPU tensor device: {cpu_tensor.device()}")
    print(f"CUDA tensor device: {cuda_tensor.device()}")

    # Test device transfer
    cpu_to_cuda = cpu_tensor.cuda()
    cuda_to_cpu = cuda_tensor.cpu()

    print(f"CPU->CUDA device: {cpu_to_cuda.device()}")
    print(f"CUDA->CPU device: {cuda_to_cpu.device()}")

    print("✓ Device operation tests passed!\n")

def main():
    """Run all tests."""
    print("Starting SmollNet comprehensive tests...\n")

    try:
        test_tensor_creation()
        test_tensor_operations()
        test_activation_functions()
        test_neural_network_layers()
        test_dense_network()
        test_optimizer()
        test_gradient_computation()
        test_device_operations()
        test_training_loop()

        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
