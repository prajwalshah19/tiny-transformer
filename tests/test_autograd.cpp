#include "tiny_transformer/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tiny_transformer;

// Helper to check if two floats are approximately equal
bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// ==================== Test Cases ====================

void test_sum_backward() {
    std::cout << "Testing sum backward..." << std::endl;
    
    // y = sum(x) where x = [1, 2, 3, 4]
    // dy/dx_i = 1 for all i
    Tensor x({4}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    
    Tensor y = x.sum();  // y = 10
    assert(approx_equal(y[0], 10.0f));
    
    y.backward();
    
    // Each element should have gradient of 1.0
    Tensor grad = x.grad();
    assert(approx_equal(grad[0], 1.0f));
    assert(approx_equal(grad[1], 1.0f));
    assert(approx_equal(grad[2], 1.0f));
    assert(approx_equal(grad[3], 1.0f));
    
    std::cout << "✓ sum backward passed" << std::endl;
}

void test_mean_backward() {
    std::cout << "Testing mean backward..." << std::endl;
    
    // y = mean(x) where x = [2, 4, 6, 8]
    // dy/dx_i = 1/n for all i
    Tensor x({4}, {2.0f, 4.0f, 6.0f, 8.0f}, true);
    
    Tensor y = x.mean();  // y = 5
    assert(approx_equal(y[0], 5.0f));
    
    y.backward();
    
    // Each element should have gradient of 1/4 = 0.25
    Tensor grad = x.grad();
    assert(approx_equal(grad[0], 0.25f));
    assert(approx_equal(grad[1], 0.25f));
    assert(approx_equal(grad[2], 0.25f));
    assert(approx_equal(grad[3], 0.25f));
    
    std::cout << "✓ mean backward passed" << std::endl;
}

void test_add_backward() {
    std::cout << "Testing add backward..." << std::endl;
    
    // z = x + y, loss = sum(z)
    // dloss/dx = 1, dloss/dy = 1
    Tensor x({3}, {1.0f, 2.0f, 3.0f}, true);
    Tensor y({3}, {4.0f, 5.0f, 6.0f}, true);
    
    Tensor z = x + y;  // z = [5, 7, 9]
    Tensor loss = z.sum();  // loss = 21
    
    assert(approx_equal(loss[0], 21.0f));
    
    loss.backward();
    
    // Both x and y should have gradients of [1, 1, 1]
    Tensor grad_x = x.grad();
    Tensor grad_y = y.grad();
    
    for (size_t i = 0; i < 3; ++i) {
        assert(approx_equal(grad_x[i], 1.0f));
        assert(approx_equal(grad_y[i], 1.0f));
    }
    
    std::cout << "✓ add backward passed" << std::endl;
}

void test_sub_backward() {
    std::cout << "Testing sub backward..." << std::endl;
    
    // z = x - y, loss = sum(z)
    // dloss/dx = 1, dloss/dy = -1
    Tensor x({3}, {5.0f, 6.0f, 7.0f}, true);
    Tensor y({3}, {1.0f, 2.0f, 3.0f}, true);
    
    Tensor z = x - y;  // z = [4, 4, 4]
    Tensor loss = z.sum();  // loss = 12
    
    assert(approx_equal(loss[0], 12.0f));
    
    loss.backward();
    
    Tensor grad_x = x.grad();
    Tensor grad_y = y.grad();
    
    for (size_t i = 0; i < 3; ++i) {
        assert(approx_equal(grad_x[i], 1.0f));
        assert(approx_equal(grad_y[i], -1.0f));
    }
    
    std::cout << "✓ sub backward passed" << std::endl;
}

void test_mul_backward() {
    std::cout << "Testing mul (element-wise) backward..." << std::endl;
    
    // z = x * y (element-wise), loss = sum(z)
    // dloss/dx_i = y_i, dloss/dy_i = x_i
    Tensor x({3}, {2.0f, 3.0f, 4.0f}, true);
    Tensor y({3}, {5.0f, 6.0f, 7.0f}, true);
    
    Tensor z = x * y;  // z = [10, 18, 28]
    Tensor loss = z.sum();  // loss = 56
    
    assert(approx_equal(loss[0], 56.0f));
    
    loss.backward();
    
    Tensor grad_x = x.grad();
    Tensor grad_y = y.grad();
    
    // grad_x should be y values: [5, 6, 7]
    assert(approx_equal(grad_x[0], 5.0f));
    assert(approx_equal(grad_x[1], 6.0f));
    assert(approx_equal(grad_x[2], 7.0f));
    
    // grad_y should be x values: [2, 3, 4]
    assert(approx_equal(grad_y[0], 2.0f));
    assert(approx_equal(grad_y[1], 3.0f));
    assert(approx_equal(grad_y[2], 4.0f));
    
    std::cout << "✓ mul backward passed" << std::endl;
}

void test_scalar_mul_backward() {
    std::cout << "Testing scalar mul backward..." << std::endl;
    
    // y = x * 3, loss = sum(y)
    // dloss/dx_i = 3
    Tensor x({3}, {1.0f, 2.0f, 3.0f}, true);
    
    Tensor y = x * 3.0f;  // y = [3, 6, 9]
    Tensor loss = y.sum();  // loss = 18
    
    assert(approx_equal(loss[0], 18.0f));
    
    loss.backward();
    
    Tensor grad_x = x.grad();
    
    // Each element should have gradient of 3.0
    assert(approx_equal(grad_x[0], 3.0f));
    assert(approx_equal(grad_x[1], 3.0f));
    assert(approx_equal(grad_x[2], 3.0f));
    
    std::cout << "✓ scalar mul backward passed" << std::endl;
}

void test_matmul_backward() {
    std::cout << "Testing matmul backward..." << std::endl;
    
    // C = A @ B where A is (2,3), B is (3,2)
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f, 
                      4.0f, 5.0f, 6.0f}, true);
    Tensor B({3, 2}, {7.0f, 8.0f, 
                      9.0f, 10.0f, 
                      11.0f, 12.0f}, true);
    
    Tensor C = A.matmul(B);  // C is (2,2)
    // C = [[58, 64], [139, 154]]
    
    assert(approx_equal(C.at({0, 0}), 58.0f));
    assert(approx_equal(C.at({0, 1}), 64.0f));
    assert(approx_equal(C.at({1, 0}), 139.0f));
    assert(approx_equal(C.at({1, 1}), 154.0f));
    
    Tensor loss = C.sum();  // loss = 58 + 64 + 139 + 154 = 415
    assert(approx_equal(loss[0], 415.0f));
    
    loss.backward();
    
    Tensor grad_A = A.grad();
    Tensor grad_B = B.grad();
    
    // dL/dA = ones(2,2) @ B^T
    // B^T = [[7, 9, 11], [8, 10, 12]]
    // dL/dA = [[15, 19, 23], [15, 19, 23]]
    assert(approx_equal(grad_A.at({0, 0}), 15.0f));
    assert(approx_equal(grad_A.at({0, 1}), 19.0f));
    assert(approx_equal(grad_A.at({0, 2}), 23.0f));
    assert(approx_equal(grad_A.at({1, 0}), 15.0f));
    assert(approx_equal(grad_A.at({1, 1}), 19.0f));
    assert(approx_equal(grad_A.at({1, 2}), 23.0f));
    
    // dL/dB = A^T @ ones(2,2)
    // A^T = [[1, 4], [2, 5], [3, 6]]
    // dL/dB = [[5, 5], [7, 7], [9, 9]]
    assert(approx_equal(grad_B.at({0, 0}), 5.0f));
    assert(approx_equal(grad_B.at({0, 1}), 5.0f));
    assert(approx_equal(grad_B.at({1, 0}), 7.0f));
    assert(approx_equal(grad_B.at({1, 1}), 7.0f));
    assert(approx_equal(grad_B.at({2, 0}), 9.0f));
    assert(approx_equal(grad_B.at({2, 1}), 9.0f));
    
    std::cout << "✓ matmul backward passed" << std::endl;
}

void test_chain_rule() {
    std::cout << "Testing chain rule (multiple ops)..." << std::endl;
    
    // y = (x * 2 + 1).sum()
    // dy/dx = 2
    Tensor x({3}, {1.0f, 2.0f, 3.0f}, true);
    Tensor ones({3}, {1.0f, 1.0f, 1.0f}, false);
    
    Tensor scaled = x * 2.0f;     // [2, 4, 6]
    Tensor shifted = scaled + ones;  // [3, 5, 7]
    Tensor loss = shifted.sum();     // 15
    
    assert(approx_equal(loss[0], 15.0f));
    
    loss.backward();
    
    Tensor grad_x = x.grad();
    
    // Gradient should be 2 for each element
    assert(approx_equal(grad_x[0], 2.0f));
    assert(approx_equal(grad_x[1], 2.0f));
    assert(approx_equal(grad_x[2], 2.0f));
    
    std::cout << "✓ chain rule passed" << std::endl;
}

void test_mse_loss() { // for OLS lin reg implementation
    std::cout << "Testing MSE loss pattern..." << std::endl;
    
    // MSE = mean((pred - target)^2)
    // loss = sum((pred - target) * (pred - target))
    // Now works correctly with topological sort!
    Tensor pred({3}, {2.0f, 4.0f, 6.0f}, true);
    Tensor target({3}, {1.0f, 3.0f, 5.0f}, false);  // target doesn't need grad
    
    Tensor diff = pred - target;  // [1, 1, 1]
    Tensor squared = diff * diff; // [1, 1, 1]
    Tensor loss = squared.sum();  // 3
    
    assert(approx_equal(loss[0], 3.0f));
    
    loss.backward();
    
    Tensor grad_pred = pred.grad();
    
    // d(sum((p-t)^2))/dp = 2*(p-t) = [2, 2, 2]
    assert(approx_equal(grad_pred[0], 2.0f));
    assert(approx_equal(grad_pred[1], 2.0f));
    assert(approx_equal(grad_pred[2], 2.0f));
    
    std::cout << "✓ MSE loss pattern passed" << std::endl;
}

void test_zero_grad() {
    std::cout << "Testing zero_grad..." << std::endl;
    
    Tensor x({2}, {1.0f, 2.0f}, true);
    
    // First backward
    Tensor y1 = x.sum();
    y1.backward();
    assert(approx_equal(x.grad()[0], 1.0f));
    
    // Second backward WITHOUT zero_grad (gradients accumulate)
    Tensor y2 = x.sum();
    y2.backward();
    assert(approx_equal(x.grad()[0], 2.0f));  // Accumulated!
    
    // Now zero the gradients
    x.zero_grad();
    
    // Third backward (fresh gradients)
    Tensor y3 = x.sum();
    y3.backward();
    assert(approx_equal(x.grad()[0], 1.0f));  // Back to 1
    
    std::cout << "✓ zero_grad passed" << std::endl;
}

// ==================== Main ====================

int main() {
    std::cout << "=== Running Autograd Tests ===" << std::endl << std::endl;
    
    try {
        test_sum_backward();
        test_mean_backward();
        test_add_backward();
        test_sub_backward();
        test_mul_backward();
        test_scalar_mul_backward();
        test_matmul_backward();
        test_chain_rule();
        test_mse_loss();
        test_zero_grad();
        
        std::cout << "\n=== All autograd tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
