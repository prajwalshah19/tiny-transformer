#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iomanip>

namespace test {

struct TestResult {
    std::string name;
    bool passed;
    std::string error_message;
    double duration_ms;
};

struct SuiteResult {
    std::string suite_name;
    std::vector<TestResult> results;
    int passed = 0;
    int failed = 0;
    double total_duration_ms = 0;
};

class TestSuite {
public:
    TestSuite(const std::string& name) : name_(name) {}
    
    void add(const std::string& test_name, std::function<void()> test_fn) {
        tests_.push_back({test_name, test_fn});
    }
    
    SuiteResult run(bool verbose = true) {
        SuiteResult suite_result;
        suite_result.suite_name = name_;
        
        if (verbose) {
            std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
            std::cout << "│ " << std::left << std::setw(60) << name_ << "│\n";
            std::cout << "└─────────────────────────────────────────────────────────────┘\n";
        }
        
        for (const auto& [test_name, test_fn] : tests_) {
            TestResult result;
            result.name = test_name;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                test_fn();
                result.passed = true;
                suite_result.passed++;
                
                if (verbose) {
                    std::cout << "  ✓ " << test_name;
                }
            } catch (const std::exception& e) {
                result.passed = false;
                result.error_message = e.what();
                suite_result.failed++;
                
                if (verbose) {
                    std::cout << "  ✗ " << test_name << "\n";
                    std::cout << "    └─ Error: " << e.what();
                }
            } catch (...) {
                result.passed = false;
                result.error_message = "Unknown exception";
                suite_result.failed++;
                
                if (verbose) {
                    std::cout << "  ✗ " << test_name << "\n";
                    std::cout << "    └─ Error: Unknown exception";
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            suite_result.total_duration_ms += result.duration_ms;
            
            if (verbose) {
                std::cout << " (" << std::fixed << std::setprecision(1) 
                          << result.duration_ms << "ms)\n";
            }
            
            suite_result.results.push_back(result);
        }
        
        return suite_result;
    }
    
private:
    std::string name_;
    std::vector<std::pair<std::string, std::function<void()>>> tests_;
};

class TestRunner {
public:
    void add_suite(std::function<SuiteResult()> suite_fn) {
        suite_fns_.push_back(suite_fn);
    }
    
    int run() {
        std::cout << "\n";
        std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              TINY TRANSFORMER TEST RUNNER                   ║\n";
        std::cout << "╚═════════════════════════════════════════════════════════════╝\n";
        
        int total_passed = 0;
        int total_failed = 0;
        double total_duration = 0;
        std::vector<SuiteResult> all_results;
        
        for (auto& suite_fn : suite_fns_) {
            SuiteResult result = suite_fn();
            all_results.push_back(result);
            total_passed += result.passed;
            total_failed += result.failed;
            total_duration += result.total_duration_ms;
        }
        
        // Summary
        std::cout << "\n";
        std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "│                        SUMMARY                              │\n";
        std::cout << "├─────────────────────────────────────────────────────────────┤\n";
        
        for (const auto& suite : all_results) {
            std::string status = (suite.failed == 0) ? "✓" : "✗";
            std::cout << "│ " << status << " " << std::left << std::setw(40) << suite.suite_name
                      << std::right << std::setw(3) << suite.passed << " passed, "
                      << std::setw(3) << suite.failed << " failed │\n";
        }
        
        std::cout << "├─────────────────────────────────────────────────────────────┤\n";
        
        // Total line
        std::string total_status = (total_failed == 0) ? "PASSED" : "FAILED";
        std::cout << "│ Total: " << std::setw(3) << total_passed << " passed, "
                  << std::setw(3) << total_failed << " failed"
                  << std::setw(20) << "" << std::fixed << std::setprecision(0) 
                  << total_duration << "ms │\n";
        std::cout << "└─────────────────────────────────────────────────────────────┘\n";
        
        if (total_failed == 0) {
            std::cout << "\n  🎉 All tests passed!\n\n";
        } else {
            std::cout << "\n  ❌ " << total_failed << " test(s) failed.\n\n";
            
            // Show failed tests
            std::cout << "Failed tests:\n";
            for (const auto& suite : all_results) {
                for (const auto& test : suite.results) {
                    if (!test.passed) {
                        std::cout << "  • " << suite.suite_name << " > " << test.name << "\n";
                        std::cout << "    " << test.error_message << "\n";
                    }
                }
            }
            std::cout << "\n";
        }
        
        return (total_failed == 0) ? 0 : 1;
    }
    
private:
    std::vector<std::function<SuiteResult()>> suite_fns_;
};

// Helper macro for assertions with better messages
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: " #condition); \
        } \
    } while(0)

#define TEST_ASSERT_MSG(condition, msg) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + msg); \
        } \
    } while(0)

#define TEST_APPROX(a, b, eps) \
    do { \
        if (std::fabs((a) - (b)) >= (eps)) { \
            throw std::runtime_error("Approx equal failed: " #a " ≈ " #b); \
        } \
    } while(0)

} // namespace test
