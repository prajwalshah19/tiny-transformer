# tiny-transformer
Learning ML and building a distributed transformer from scratch in C++.

## Building

```bash
make          # Compile source files
make clean    # Remove build artifacts
```

## Running Tests

```bash
# Run all tests (unified runner)
make test

# Run specific test suite
make test-tensor
make test-autograd
make test-linear_regression
```

## Project Structure

```
tiny-transformer/
├── include/tiny_transformer/
│   ├── tensor.hpp              # Tensor class with autograd
│   └── linear_regression.hpp   # OLS linear regression
├── src/
│   ├── tensor.cpp
│   └── linear_regression.cpp
├── tests/
│   ├── run_tests.cpp           # Unified test runner (source of truth)
│   ├── test_tensor.cpp         # Individual tests (for debugging)
│   ├── test_autograd.cpp
│   └── test_linear_regression.cpp
└── Makefile
```

## Features

- **Tensor**: N-dimensional array with broadcasting support
- **Autograd**: Reverse-mode automatic differentiation
- **Linear Regression**: OLS with gradient descent, fit/refit/predict API
