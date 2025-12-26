// Unified test runner
// Includes all test files and runs them with nice output

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iomanip>

// ============================================================
// Test Framework
// ============================================================

namespace test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

struct TestResult {
    std::string suite;
    std::string name;
    bool passed;
    std::string error;
    double duration_ms;
};

class TestRunner {
public:
    static TestRunner& instance() {
        static TestRunner runner;
        return runner;
    }
    
    void add(const std::string& suite, const std::string& name, std::function<void()> fn) {
        tests_.push_back({suite, name, fn});
    }
    
    int run() {
        std::cout << "\n";
        std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              TINY TRANSFORMER TEST RUNNER                   ║\n";
        std::cout << "╚═════════════════════════════════════════════════════════════╝\n";
        
        std::vector<TestResult> results;
        std::string current_suite;
        int passed = 0, failed = 0;
        
        for (const auto& test : tests_) {
            if (test.suite != current_suite) {
                current_suite = test.suite;
                std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
                std::cout << "│ " << std::left << std::setw(60) << current_suite << "│\n";
                std::cout << "└─────────────────────────────────────────────────────────────┘\n";
            }
            
            TestResult result;
            result.suite = test.suite;
            result.name = test.name;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                test.fn();
                result.passed = true;
                passed++;
                std::cout << "  ✓ " << test.name;
            } catch (const std::exception& e) {
                result.passed = false;
                result.error = e.what();
                failed++;
                std::cout << "  ✗ " << test.name << "\n";
                std::cout << "    └─ " << e.what();
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            std::cout << " (" << std::fixed << std::setprecision(1) << result.duration_ms << "ms)\n";
            results.push_back(result);
        }
        
        // Summary
        std::cout << "\n";
        std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "│                        SUMMARY                              │\n";
        std::cout << "├─────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Total: " << std::setw(3) << passed << " passed, "
                  << std::setw(3) << failed << " failed"
                  << std::setw(35) << "" << "│\n";
        std::cout << "└─────────────────────────────────────────────────────────────┘\n";
        
        if (failed == 0) {
            std::cout << "\n  🎉 All tests passed!\n\n";
        } else {
            std::cout << "\n  ❌ " << failed << " test(s) failed:\n";
            for (const auto& r : results) {
                if (!r.passed) {
                    std::cout << "     • " << r.suite << " > " << r.name << "\n";
                }
            }
            std::cout << "\n";
        }
        
        return failed == 0 ? 0 : 1;
    }
    
private:
    std::vector<TestCase> tests_;
};

// Helper macro for assertions
#define TEST_ASSERT(cond) \
    do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while(0)

// Registration macro
#define TEST(suite, name) \
    void test_##suite##_##name(); \
    namespace { \
        struct Register_##suite##_##name { \
            Register_##suite##_##name() { \
                test::TestRunner::instance().add(#suite, #name, test_##suite##_##name); \
            } \
        } _reg_##suite##_##name; \
    } \
    void test_##suite##_##name()

} // namespace test

// ============================================================
// Include all test files (they use TEST macro to register)
// ============================================================

// Must undef main from test files since we define our own
#define SKIP_MAIN

#include "tiny_transformer/tensor.hpp"
#include "tiny_transformer/linear_regression.hpp"
#include <cassert>
#include <cmath>

using namespace tiny_transformer;
using namespace test;

// ============================================================
// Tensor Tests
// ============================================================

TEST(Tensor, constructors) {
    Tensor scalar;
    TEST_ASSERT(scalar.size() == 1);
    TEST_ASSERT(scalar.ndim() == 0);
    
    Tensor t1({2, 3});
    TEST_ASSERT(t1.size() == 6);
    TEST_ASSERT(t1.ndim() == 2);
    
    Tensor t2({2, 2}, 5.0f);
    TEST_ASSERT(t2[0] == 5.0f);
    
    Tensor t3({2, 2}, {1, 2, 3, 4});
    TEST_ASSERT(t3[3] == 4.0f);
}

TEST(Tensor, indexing) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    TEST_ASSERT(t[0] == 1.0f);
    TEST_ASSERT(t.at({1, 2}) == 6.0f);
    t.at({0, 1}) = 99.0f;
    TEST_ASSERT(t.at({0, 1}) == 99.0f);
}

TEST(Tensor, reshape) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor reshaped = t.reshape({3, 2});
    TEST_ASSERT(reshaped.shape()[0] == 3);
    TEST_ASSERT(reshaped.shape()[1] == 2);
}

TEST(Tensor, transpose) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tr = t.transpose();
    TEST_ASSERT(tr.shape()[0] == 3);
    TEST_ASSERT(tr.shape()[1] == 2);
    TEST_ASSERT(tr.at({0, 1}) == 4.0f);
}

TEST(Tensor, element_wise_ops) {
    Tensor t1({2, 2}, {1, 2, 3, 4});
    Tensor t2({2, 2}, {5, 6, 7, 8});
    
    Tensor sum = t1 + t2;
    TEST_ASSERT(sum[0] == 6.0f);
    
    Tensor diff = t2 - t1;
    TEST_ASSERT(diff[0] == 4.0f);
    
    Tensor prod = t1 * t2;
    TEST_ASSERT(prod[0] == 5.0f);
    
    Tensor scaled = t1 * 2.0f;
    TEST_ASSERT(scaled[0] == 2.0f);
}

TEST(Tensor, matmul) {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor result = a.matmul(b);
    TEST_ASSERT(result.at({0, 0}) == 58.0f);
    TEST_ASSERT(result.at({1, 1}) == 154.0f);
}

TEST(Tensor, static_methods) {
    Tensor zeros = Tensor::zeros({2, 3});
    TEST_ASSERT(zeros[0] == 0.0f);
    
    Tensor ones = Tensor::ones({2, 2});
    TEST_ASSERT(ones[0] == 1.0f);
    
    Tensor randn = Tensor::randn({3, 3});
    TEST_ASSERT(randn.size() == 9);
}

TEST(Tensor, broadcast_add) {
    Tensor X({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor bias({1}, std::vector<float>{10.0f});
    Tensor result = X.add_broadcast(bias);
    TEST_ASSERT(result.at({0, 0}) == 11.0f);
    TEST_ASSERT(result.at({2, 1}) == 16.0f);
}

// ============================================================
// Autograd Tests
// ============================================================

static bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

TEST(Autograd, sum_backward) {
    Tensor x({4}, {1, 2, 3, 4}, true);
    Tensor y = x.sum();
    y.backward();
    Tensor grad = x.grad();
    TEST_ASSERT(approx_eq(grad[0], 1.0f));
    TEST_ASSERT(approx_eq(grad[3], 1.0f));
}

TEST(Autograd, mean_backward) {
    Tensor x({4}, {2, 4, 6, 8}, true);
    Tensor y = x.mean();
    y.backward();
    Tensor grad = x.grad();
    TEST_ASSERT(approx_eq(grad[0], 0.25f));
}

TEST(Autograd, add_backward) {
    Tensor x({3}, {1, 2, 3}, true);
    Tensor y({3}, {4, 5, 6}, true);
    Tensor z = x + y;
    z.sum().backward();
    TEST_ASSERT(approx_eq(x.grad()[0], 1.0f));
    TEST_ASSERT(approx_eq(y.grad()[0], 1.0f));
}

TEST(Autograd, mul_backward) {
    Tensor x({3}, {2, 3, 4}, true);
    Tensor y({3}, {5, 6, 7}, true);
    Tensor z = x * y;
    z.sum().backward();
    TEST_ASSERT(approx_eq(x.grad()[0], 5.0f));  // grad_x = y
    TEST_ASSERT(approx_eq(y.grad()[0], 2.0f));  // grad_y = x
}

TEST(Autograd, matmul_backward) {
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    Tensor B({3, 2}, {7, 8, 9, 10, 11, 12}, true);
    Tensor C = A.matmul(B);
    C.sum().backward();
    TEST_ASSERT(approx_eq(A.grad().at({0, 0}), 15.0f));
    TEST_ASSERT(approx_eq(B.grad().at({0, 0}), 5.0f));
}

TEST(Autograd, mse_loss) {
    Tensor pred({3}, {2, 4, 6}, true);
    Tensor target({3}, {1, 3, 5}, false);
    Tensor diff = pred - target;
    Tensor loss = (diff * diff).sum();
    loss.backward();
    TEST_ASSERT(approx_eq(pred.grad()[0], 2.0f));
}

TEST(Autograd, zero_grad) {
    Tensor x({2}, {1, 2}, true);
    x.sum().backward();
    TEST_ASSERT(approx_eq(x.grad()[0], 1.0f));
    x.sum().backward();
    TEST_ASSERT(approx_eq(x.grad()[0], 2.0f));  // accumulated
    x.zero_grad();
    x.sum().backward();
    TEST_ASSERT(approx_eq(x.grad()[0], 1.0f));  // reset
}

// ============================================================
// Linear Regression Tests
// ============================================================

TEST(LinearRegression, simple_linear) {
    size_t n = 100;
    Tensor X({n, 1}), y({n, 1});
    for (size_t i = 0; i < n; ++i) {
        float x_val = static_cast<float>(i) / 100.0f;
        X[i] = x_val;
        y[i] = 2.0f * x_val + 1.0f;
    }
    
    LinearRegression model(0.5f, 1e-7f);
    OLSResult result = model.fit(X, y, 10000);
    
    TEST_ASSERT(std::abs(result.weights[0] - 2.0f) < 0.1f);
    TEST_ASSERT(std::abs(result.bias - 1.0f) < 0.1f);
    TEST_ASSERT(result.r_squared > 0.99f);
}

TEST(LinearRegression, multivariate) {
    size_t n = 100;
    Tensor X({n, 2}), y({n, 1});
    for (size_t i = 0; i < n; ++i) {
        float x1 = static_cast<float>(i % 10) / 10.0f;
        float x2 = static_cast<float>(i / 10) / 10.0f;
        X.at({i, 0}) = x1;
        X.at({i, 1}) = x2;
        y[i] = 2.0f * x1 + 3.0f * x2 + 1.0f;
    }
    
    LinearRegression model(0.5f, 1e-7f);
    OLSResult result = model.fit(X, y, 10000);
    
    TEST_ASSERT(std::abs(result.weights[0] - 2.0f) < 0.2f);
    TEST_ASSERT(std::abs(result.weights[1] - 3.0f) < 0.2f);
    TEST_ASSERT(std::abs(result.bias - 1.0f) < 0.2f);
}

TEST(LinearRegression, predict) {
    Tensor X({50, 1}), y({50, 1});
    for (size_t i = 0; i < 50; ++i) {
        float x_val = static_cast<float>(i) / 50.0f;
        X[i] = x_val;
        y[i] = x_val;
    }
    
    LinearRegression model(0.5f, 1e-7f);
    model.fit(X, y, 5000);
    
    Tensor X_test({3, 1}, {0.2f, 0.4f, 0.6f});
    Tensor preds = model.predict(X_test);
    
    TEST_ASSERT(std::abs(preds[0] - 0.2f) < 0.05f);
    TEST_ASSERT(std::abs(preds[1] - 0.4f) < 0.05f);
}

TEST(LinearRegression, static_predict) {
    OLSResult r;
    r.weights = Tensor({2, 1}, {2.0f, 3.0f});
    r.bias = 1.0f;
    
    Tensor X_test({2, 2}, {1, 1, 2, 2});
    Tensor preds = LinearRegression::predict(X_test, r);
    
    TEST_ASSERT(std::abs(preds[0] - 6.0f) < 1e-5f);
    TEST_ASSERT(std::abs(preds[1] - 11.0f) < 1e-5f);
}

TEST(LinearRegression, refit) {
    Tensor X1({20, 1}), y1({20, 1});
    for (size_t i = 0; i < 20; ++i) {
        float x_val = static_cast<float>(i) / 20.0f;
        X1[i] = x_val;
        y1[i] = 2.0f * x_val + 1.0f;
    }
    
    LinearRegression model(0.5f, 1e-7f);
    model.fit(X1, y1, 5000);
    
    Tensor X2({20, 1}), y2({20, 1});
    for (size_t i = 0; i < 20; ++i) {
        float x_val = static_cast<float>(i + 5) / 25.0f;
        X2[i] = x_val;
        y2[i] = 2.0f * x_val + 1.0f;
    }
    
    OLSResult result = model.refit(X2, y2, 5000);
    
    TEST_ASSERT(std::abs(result.weights[0] - 2.0f) < 0.2f);
    TEST_ASSERT(std::abs(result.bias - 1.0f) < 0.2f);
}

TEST(LinearRegression, unfitted_errors) {
    LinearRegression model;
    Tensor X({5, 2}), x({2});
    
    bool threw = false;
    try { model.predict(X); } catch (...) { threw = true; }
    TEST_ASSERT(threw);
    
    threw = false;
    try { model.predict_one(x); } catch (...) { threw = true; }
    TEST_ASSERT(threw);
}

// ============================================================
// Main
// ============================================================

int main() {
    return test::TestRunner::instance().run();
}
