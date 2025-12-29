#include "tiny_transformer/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tiny_transformer;

void test_constructors() {
    std::cout << "Testing constructors..." << std::endl;
    
    // Default constructor (scalar)
    Tensor scalar;
    assert(scalar.size() == 1);
    assert(scalar.ndim() == 0);
    assert(scalar[0] == 0.0f);
    
    // Shape constructor
    Tensor t1({2, 3});
    assert(t1.size() == 6);
    assert(t1.ndim() == 2);
    assert(t1.shape()[0] == 2);
    assert(t1.shape()[1] == 3);
    
    // Fill constructor
    Tensor t2({2, 2}, 5.0f);
    assert(t2[0] == 5.0f);
    assert(t2[3] == 5.0f);
    
    // Data constructor
    std::vector<float> data = {1, 2, 3, 4};
    Tensor t3({2, 2}, data);
    assert(t3[0] == 1.0f);
    assert(t3[3] == 4.0f);
    
    std::cout << "✓ Constructors passed" << std::endl;
}

void test_indexing() {
    std::cout << "Testing indexing..." << std::endl;
    
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    
    // Flat indexing
    assert(t[0] == 1.0f);
    assert(t[5] == 6.0f);
    
    // Multi-dimensional indexing
    assert(t.at({0, 0}) == 1.0f);
    assert(t.at({0, 2}) == 3.0f);
    assert(t.at({1, 0}) == 4.0f);
    assert(t.at({1, 2}) == 6.0f);
    
    // Modification
    t.at({0, 1}) = 99.0f;
    assert(t.at({0, 1}) == 99.0f);
    
    std::cout << "✓ Indexing passed" << std::endl;
}

void test_reshape() {
    std::cout << "Testing reshape..." << std::endl;
    
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor reshaped = t.reshape({3, 2});
    
    assert(reshaped.shape()[0] == 3);
    assert(reshaped.shape()[1] == 2);
    assert(reshaped[0] == 1.0f);
    assert(reshaped[5] == 6.0f);
    assert(reshaped.at({0, 0}) == 1.0f);
    assert(reshaped.at({2, 1}) == 6.0f);
    
    std::cout << "✓ Reshape passed" << std::endl;
}

void test_transpose() {
    std::cout << "Testing transpose..." << std::endl;
    
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    // Original: [[1, 2, 3],
    //            [4, 5, 6]]
    
    Tensor transposed = t.transpose();
    // Expected: [[1, 4],
    //            [2, 5],
    //            [3, 6]]
    
    assert(transposed.shape()[0] == 3);
    assert(transposed.shape()[1] == 2);
    assert(transposed.at({0, 0}) == 1.0f);
    assert(transposed.at({0, 1}) == 4.0f);
    assert(transposed.at({1, 0}) == 2.0f);
    assert(transposed.at({2, 1}) == 6.0f);
    
    std::cout << "✓ Transpose passed" << std::endl;
}

void test_element_wise_ops() {
    std::cout << "Testing element-wise operations..." << std::endl;
    
    Tensor t1({2, 2}, {1, 2, 3, 4});
    Tensor t2({2, 2}, {5, 6, 7, 8});
    
    // Addition
    Tensor sum = t1 + t2;
    assert(sum[0] == 6.0f);
    assert(sum[3] == 12.0f);
    
    // Subtraction
    Tensor diff = t2 - t1;
    assert(diff[0] == 4.0f);
    assert(diff[3] == 4.0f);
    
    // Element-wise multiplication
    Tensor prod = t1 * t2;
    assert(prod[0] == 5.0f);
    assert(prod[3] == 32.0f);
    
    // Scalar multiplication - BUG FIX NEEDED
    Tensor scaled = t1 * 2.0f;
    assert(scaled[0] == 2.0f);  // Will fail - you have + instead of *
    assert(scaled[3] == 8.0f);
    
    std::cout << "✓ Element-wise operations passed" << std::endl;
}

void test_matmul() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    // 2x3 * 3x2 = 2x2
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    
    Tensor result = a.matmul(b);
    
    assert(result.shape()[0] == 2);
    assert(result.shape()[1] == 2);
    
    // Manual calculation:
    // [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //  [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    
    assert(result.at({0, 0}) == 58.0f);
    assert(result.at({0, 1}) == 64.0f);
    assert(result.at({1, 0}) == 139.0f);
    assert(result.at({1, 1}) == 154.0f);
    
    std::cout << "✓ Matrix multiplication passed" << std::endl;
}

void test_static_methods() {
    std::cout << "Testing static factory methods..." << std::endl;
    
    // Zeros
    Tensor zeros = Tensor::zeros({2, 3});
    assert(zeros.size() == 6);
    assert(zeros[0] == 0.0f);
    assert(zeros[5] == 0.0f);
    
    // Ones
    Tensor ones = Tensor::ones({2, 2});
    assert(ones.size() == 4);
    assert(ones[0] == 1.0f);
    assert(ones[3] == 1.0f);
    
    // Randn (just check shape and size)
    Tensor randn = Tensor::randn({3, 3});
    assert(randn.size() == 9);
    assert(randn.ndim() == 2);
    
    std::cout << "✓ Static factory methods passed" << std::endl;
}

void test_fill() {
    std::cout << "Testing fill..." << std::endl;
    
    Tensor t({2, 2});
    t.fill(7.5f);
    
    assert(t[0] == 7.5f);
    assert(t[3] == 7.5f);
    
    std::cout << "✓ Fill passed" << std::endl;
}

void test_broadcast_add() {
    std::cout << "Testing broadcast addition..." << std::endl;
    
    // Test 1: Bias-style broadcast - (3, 2) + (1,) -> (3, 2)
    Tensor X({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor bias({1}, std::vector<float>{10.0f});
    
    Tensor result1 = X.add_broadcast(bias);
    assert(result1.shape()[0] == 3);
    assert(result1.shape()[1] == 2);
    assert(result1.at({0, 0}) == 11.0f);  // 1 + 10
    assert(result1.at({0, 1}) == 12.0f);  // 2 + 10
    assert(result1.at({2, 1}) == 16.0f);  // 6 + 10
    
    std::cout << "  ✓ Scalar bias broadcast passed" << std::endl;
    
    // Test 2: Row broadcast - (3, 2) + (2,) -> (3, 2)
    Tensor row({2}, {100.0f, 200.0f});
    Tensor result2 = X.add_broadcast(row);
    assert(result2.shape()[0] == 3);
    assert(result2.shape()[1] == 2);
    assert(result2.at({0, 0}) == 101.0f);  // 1 + 100
    assert(result2.at({0, 1}) == 202.0f);  // 2 + 200
    assert(result2.at({1, 0}) == 103.0f);  // 3 + 100
    assert(result2.at({2, 1}) == 206.0f);  // 6 + 200
    
    std::cout << "  ✓ Row vector broadcast passed" << std::endl;
    
    // Test 3: Incompatible shapes - (3, 2) + (3,) should throw
    Tensor incompatible({3}, {1, 2, 3});
    bool threw = false;
    try {
        Tensor bad = X.add_broadcast(incompatible);
    } catch (const std::invalid_argument& e) {
        threw = true;
    }
    assert(threw);
    
    std::cout << "  ✓ Incompatible broadcast throws correctly" << std::endl;
    
    std::cout << "✓ Broadcast addition passed" << std::endl;
}

void test_broadcast_autograd() {
    std::cout << "Testing broadcast autograd..." << std::endl;
    
    // (3, 2) + (1,) with gradients
    Tensor X({3, 2}, {1, 2, 3, 4, 5, 6}, true);
    Tensor bias({1}, {0.0f}, true);
    
    Tensor result = X.add_broadcast(bias);
    Tensor loss = result.sum();
    
    loss.backward();
    
    // d(sum)/dX_i = 1 for all i
    Tensor x_grad = X.grad();
    for (size_t i = 0; i < X.size(); ++i) {
        assert(std::abs(x_grad[i] - 1.0f) < 1e-5f);
    }
    
    // d(sum)/dbias = 6 (gradient accumulated from all 6 elements)
    Tensor b_grad = bias.grad();
    assert(std::abs(b_grad[0] - 6.0f) < 1e-5f);
    
    std::cout << "✓ Broadcast autograd passed" << std::endl;
}

void test_sigmoid() {
    std::cout << "Testing sigmoid..." << std::endl;
    Tensor t({3}, {-1.0f, 0.0f, 1.0f});
    Tensor out = t.sigmoid();
    assert(std::abs(out[0] - 0.268941f) < 1e-5f); // sigmoid(-1)
    assert(std::abs(out[1] - 0.5f) < 1e-5f);      // sigmoid(0)
    assert(std::abs(out[2] - 0.731058f) < 1e-5f); // sigmoid(1)
    std::cout << "\u2713 Sigmoid passed" << std::endl;
}

void test_relu() {
    std::cout << "Testing relu..." << std::endl;
    Tensor t({4}, {-2.0f, 0.0f, 3.5f, -0.1f});
    Tensor out = t.relu();
    assert(out[0] == 0.0f);
    assert(out[1] == 0.0f);
    assert(out[2] == 3.5f);
    assert(out[3] == 0.0f);
    std::cout << "\u2713 ReLU passed" << std::endl;
}

void test_softmax() {
    std::cout << "Testing softmax..." << std::endl;
    Tensor t({3}, {1.0f, 2.0f, 3.0f});
    Tensor out = t.softmax();
    float sum = out[0] + out[1] + out[2];
    assert(std::abs(sum - 1.0f) < 1e-5f);
    // Check values (softmax should be monotonic)
    assert(out[0] < out[1] && out[1] < out[2]);
    std::cout << "\u2713 Softmax passed" << std::endl;
}

void test_print() {
    std::cout << "\nTesting print (visual inspection):" << std::endl;
    
    std::cout << "Scalar:" << std::endl;
    Tensor scalar;
    scalar.print();
    
    std::cout << "\n1D vector:" << std::endl;
    Tensor vec({5}, {1, 2, 3, 4, 5});
    vec.print();
    
    std::cout << "\n2D matrix:" << std::endl;
    Tensor mat({2, 3}, {1, 2, 3, 4, 5, 6});
    mat.print();
    
    std::cout << "\n3D tensor:" << std::endl;
    Tensor tensor3d({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    tensor3d.print();
}

int main() {
    std::cout << "=== Running Tensor Tests ===" << std::endl << std::endl;
    
    try {
        test_constructors();
        test_indexing();
        test_reshape();
        test_transpose();
        test_element_wise_ops();
        test_matmul();
        test_static_methods();
        test_fill();
        test_print();
        
        std::cout << "\n=== All tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}