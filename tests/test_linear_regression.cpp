#include "../include/tiny_transformer/linear_regression.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tiny_transformer;

void test_simple_linear() {
    std::cout << "Testing simple linear regression (y = 2x + 1)..." << std::endl;
    
    // Generate data: y = 2*x + 1
    size_t n_samples = 100;
    Tensor X({n_samples, 1});
    Tensor y({n_samples, 1});
    
    for (size_t i = 0; i < n_samples; ++i) {
        float x_val = static_cast<float>(i) / 100.0f;  // [0.00, 0.99]
        X[i] = x_val;
        y[i] = 2.0f * x_val + 1.0f;  // y = 2*x + 1
    }
    
    LinearRegression model(0.5f, 1e-7f);
    OLSResult result = model.fit(X, y, 10000);
    
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << std::endl;
    std::cout << "  Weight: " << result.weights[0] << " (expected: 2.0)" << std::endl;
    std::cout << "  Bias: " << result.bias << " (expected: 1.0)" << std::endl;
    std::cout << "  R²: " << result.r_squared << std::endl;
    std::cout << "  MSE: " << result.mse << std::endl;
    
    assert(std::abs(result.weights[0] - 2.0f) < 0.1f);
    assert(std::abs(result.bias - 1.0f) < 0.1f);
    assert(result.r_squared > 0.99f);
    
    std::cout << "✓ Simple linear regression passed" << std::endl;
}

void test_multivariate() {
    std::cout << "Testing multivariate regression (y = 2*x1 + 3*x2 + 1)..." << std::endl;
    
    size_t n_samples = 100;
    Tensor X({n_samples, 2});
    Tensor y({n_samples, 1});
    
    for (size_t i = 0; i < n_samples; ++i) {
        float x1 = static_cast<float>(i % 10) / 10.0f;  // [0.0, 0.9]
        float x2 = static_cast<float>(i / 10) / 10.0f;  // [0.0, 0.9]
        X.at({i, 0}) = x1;
        X.at({i, 1}) = x2;
        y[i] = 2.0f * x1 + 3.0f * x2 + 1.0f;
    }
    
    LinearRegression model(0.5f, 1e-7f);
    OLSResult result = model.fit(X, y, 10000);
    
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << std::endl;
    std::cout << "  Weight[0]: " << result.weights[0] << " (expected: 2.0)" << std::endl;
    std::cout << "  Weight[1]: " << result.weights[1] << " (expected: 3.0)" << std::endl;
    std::cout << "  Bias: " << result.bias << " (expected: 1.0)" << std::endl;
    std::cout << "  R²: " << result.r_squared << std::endl;
    
    assert(std::abs(result.weights[0] - 2.0f) < 0.2f);
    assert(std::abs(result.weights[1] - 3.0f) < 0.2f);
    assert(std::abs(result.bias - 1.0f) < 0.2f);
    assert(result.r_squared > 0.95f);
    
    std::cout << "✓ Multivariate regression passed" << std::endl;
}

void test_predict() {
    std::cout << "Testing predict functions..." << std::endl;
    
    // y = x, normalized range
    Tensor X({50, 1});
    Tensor y({50, 1});
    for (size_t i = 0; i < 50; ++i) {
        float x_val = static_cast<float>(i) / 50.0f;  // [0.0, 0.98]
        X[i] = x_val;
        y[i] = x_val;  // y = x
    }
    
    LinearRegression model(0.5f, 1e-7f);
    model.fit(X, y, 5000);
    
    // Test batch predict (within training range)
    Tensor X_test({3, 1}, {0.2f, 0.4f, 0.6f});
    Tensor preds = model.predict(X_test);
    
    std::cout << "  Predictions: " << preds[0] << ", " << preds[1] << ", " << preds[2] << std::endl;
    
    assert(preds.shape()[0] == 3);
    assert(std::abs(preds[0] - 0.2f) < 0.05f);
    assert(std::abs(preds[1] - 0.4f) < 0.05f);
    assert(std::abs(preds[2] - 0.6f) < 0.05f);
    
    // Test single predict
    Tensor x_single({1}, std::vector<float>{0.3f});
    float pred_single = model.predict_one(x_single);
    assert(std::abs(pred_single - 0.3f) < 0.05f);
    
    std::cout << "✓ Predict functions passed" << std::endl;
}

void test_static_predict() {
    std::cout << "Testing static predict API..." << std::endl;
    
    // Create a known result manually
    OLSResult manual_result;
    manual_result.weights = Tensor({2, 1}, {2.0f, 3.0f});
    manual_result.bias = 1.0f;
    
    // y = 2*x1 + 3*x2 + 1
    Tensor X_test({2, 2}, {1.0f, 1.0f,   // 2*1 + 3*1 + 1 = 6
                          2.0f, 2.0f});  // 2*2 + 3*2 + 1 = 11
    
    Tensor preds = LinearRegression::predict(X_test, manual_result);
    
    assert(std::abs(preds[0] - 6.0f) < 1e-5f);
    assert(std::abs(preds[1] - 11.0f) < 1e-5f);
    
    // Test predict_one
    Tensor x_single({2}, {1.0f, 2.0f});  // 2*1 + 3*2 + 1 = 9
    float pred = LinearRegression::predict_one(x_single, manual_result);
    assert(std::abs(pred - 9.0f) < 1e-5f);
    
    std::cout << "✓ Static predict API passed" << std::endl;
}

void test_refit() {
    std::cout << "Testing refit (warm start)..." << std::endl;
    
    // Initial fit: y = 2*x + 1, normalized
    Tensor X1({20, 1});
    Tensor y1({20, 1});
    for (size_t i = 0; i < 20; ++i) {
        float x_val = static_cast<float>(i) / 20.0f;  // [0.0, 0.95]
        X1[i] = x_val;
        y1[i] = 2.0f * x_val + 1.0f;  // y = 2*x + 1
    }
    
    LinearRegression model(0.5f, 1e-7f);
    OLSResult result1 = model.fit(X1, y1, 5000);
    
    std::cout << "  Initial - W: " << result1.weights[0] << ", b: " << result1.bias << std::endl;
    
    // Refit on similar data (same relationship, shifted range)
    Tensor X2({20, 1});
    Tensor y2({20, 1});
    for (size_t i = 0; i < 20; ++i) {
        float x_val = static_cast<float>(i + 5) / 25.0f;  // [0.2, 0.96]
        X2[i] = x_val;
        y2[i] = 2.0f * x_val + 1.0f;  // same: y = 2*x + 1
    }
    
    OLSResult result2 = model.refit(X2, y2, 5000);
    
    std::cout << "  Initial fit iterations: " << result1.iterations << std::endl;
    std::cout << "  Refit iterations: " << result2.iterations << std::endl;
    std::cout << "  Refit - W: " << result2.weights[0] << ", b: " << result2.bias << std::endl;
    
    assert(std::abs(result2.weights[0] - 2.0f) < 0.2f);
    assert(std::abs(result2.bias - 1.0f) < 0.2f);
    
    std::cout << "✓ Refit passed" << std::endl;
}

void test_unfitted_errors() {
    std::cout << "Testing unfitted model errors..." << std::endl;
    
    LinearRegression model;
    Tensor X({5, 2});
    Tensor x({2});
    
    bool threw_predict = false;
    try {
        model.predict(X);
    } catch (const std::runtime_error&) {
        threw_predict = true;
    }
    assert(threw_predict);
    
    bool threw_predict_one = false;
    try {
        model.predict_one(x);
    } catch (const std::runtime_error&) {
        threw_predict_one = true;
    }
    assert(threw_predict_one);
    
    bool threw_result = false;
    try {
        model.result();
    } catch (const std::runtime_error&) {
        threw_result = true;
    }
    assert(threw_result);
    
    std::cout << "✓ Unfitted error handling passed" << std::endl;
}

int main() {
    std::cout << std::unitbuf;  // unbuffered output
    std::cout << "=== Running Linear Regression Tests ===" << std::endl << std::endl;
    
    try {
        test_simple_linear();
        test_multivariate();
        test_predict();
        test_static_predict();
        test_refit();
        test_unfitted_errors();
        
        std::cout << "\n=== All linear regression tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}