#include "../include/tiny_transformer/tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

namespace tiny_transformer {

void Tensor::compute_strides() {
    if (shape_.empty()) {
        strides_.clear(); // scalar
        return;
    }
    strides_.resize(shape_.size());
    size_t val = 1;
    for (int i = static_cast<int>(shape_.size() - 1); i >= 0; --i) {
        strides_[i] = val;
        val *= shape_[i];
    }
}

size_t Tensor::compute_offset(const std::vector<size_t>& indices) const {
    if (shape_.size() != indices.size()) {
        throw std::out_of_range("Index dimensions don't match tensor dimensions");
    } // invalid indices
    size_t offset = 0;
    for (size_t i = 0; i < shape_.size(); i++) {
        offset += (strides_[i] * indices[i]);
    }
    return offset;
}

void Tensor::check_shape_compatibility(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch");
    }
}

Tensor::Tensor() {
    // default value is a scalar of value 0
    shape_ = {};
    compute_strides();
    data_ = {0.0f};
}

Tensor::Tensor(const std::vector<size_t>& shape) {
    shape_ = shape;
    compute_strides();
    size_t total_size = 1;
    // how many entries do we need
    for (size_t dim : shape_) {
        total_size *= dim;
    }
    // fill with 0s
    data_.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float fill) {
    shape_ = shape;
    compute_strides();
    size_t total_size = 1;
    // how many entries do we need
    for (size_t dim : shape_) {
        total_size *= dim;
    }
    // fill with given value
    data_.resize(total_size, fill);
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data) {
    shape_ = shape;
    compute_strides();
    size_t total_size = 1;
    // how many entries do we need
    for (size_t dim : shape_) {
        total_size *= dim;
    }
    if (data.size() != total_size) {
        throw std::invalid_argument("Data size doesn't match shape");
    }
    data_ = data;
}

float& Tensor::operator[](size_t idx) {
    return data_[idx];
}

const float& Tensor::operator[](size_t idx) const {
    return data_[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Index dimensions don't match tensor dimensions");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }

    size_t offset = compute_offset(indices);
    return data_[offset];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Index dimensions don't match tensor dimensions");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }

    size_t offset = compute_offset(indices);
    return data_[offset];
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t total_size = 1;
    for (size_t dim : new_shape) {
        total_size *= dim;
    }
    if (total_size != data_.size()) {
        throw std::invalid_argument("New shape must have same total size as original");
    }
    return Tensor(new_shape, data_);
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Shape must be 2d to transpose");
    }

    size_t rows = shape_[0];
    size_t cols = shape_[1];
    
    Tensor result({cols, rows});

    // swap along axis
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at({j, i}) = this->at({i, j});
        }
    }

    return result;

}

Tensor Tensor::operator+(const Tensor& other) const {
    check_shape_compatibility(other);

    // do the math
    std::vector<float> result_data(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result_data[i] = data_[i] + other.data_[i];
    }

    // create result as shared pointer
    auto result = std::make_shared<Tensor>(shape_, result_data);

    if (requires_grad_ || other.requires_grad_) {
        result->requires_grad_ = true;
        
        // store left and right as shared pointers as parents of result
        auto left = std::make_shared<Tensor>(*this);
        auto right = std::make_shared<Tensor>(other);
        result->parents_ = {left, right};
        
        // capture result as weak_ptr to avoid circular reference
        std::weak_ptr<Tensor> result_weak = result;
        
        result->backward_fn_ = [result_weak, left, right]() {
            auto res = result_weak.lock();
            if (!res || !res->grad_) return;
            
            // results current grad
            const Tensor& grad = *(res->grad_);
            
            // accumulate parent grads with it
            if (left->requires_grad_) {
                left->backward(grad);
            }
            if (right->requires_grad_) {
                right->backward(grad);
            }
        };
    }
    
    return *result; // return cpy
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_shape_compatibility(other);
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatibility(other);
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // 2d vectors at a minimum
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    // inner dimensions must match
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matmul");
    }

    size_t m = shape_[0];      // rows of this
    size_t n = shape_[1];      // cols of this = rows of other
    size_t p = other.shape_[1]; // cols of other

    Tensor result({m, p});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += this->at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }

    return result;
}

void Tensor::fill(float value) {
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = value;
    }
}

void Tensor::print() const {
    if (shape_.empty()) {
        // Scalar
        std::cout << "Scalar: " << data_[0] << std::endl;
        return;
    }
    
    if (shape_.size() == 1) {
        // 1D vector
        std::cout << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            std::cout << data_[i];
            if (i < shape_[0] - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        return;
    }
    
    if (shape_.size() == 2) {
        // 2D matrix
        std::cout << "[" << std::endl;
        for (size_t i = 0; i < shape_[0]; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < shape_[1]; ++j) {
                std::cout << this->at({i, j});
                if (j < shape_[1] - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i < shape_[0] - 1) std::cout << ",";
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
        return;
    }
    
    // Higher dimensions - simple flat print
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Data: [";
    for (size_t i = 0; i < data_.size(); ++i) {
        std::cout << data_[i];
        if (i < data_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::randn(const std::vector<size_t>& shape) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }

    // generate random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        // generate a random value from our dist
        data[i] = dist(gen);
    }
    
    return Tensor(shape, data);
}

// AUTOGRAD Implementation

void Tensor::accumulate_grad(const Tensor& incoming_grad) {
    if (shape_ != incoming_grad.shape_) {
        throw std::runtime_error("Gradient shape mismatch");
    }

    if (!grad_) { // init gradient
        grad_ = std::make_shared<Tensor>(shape_, 0.0f);
    }

    // accumulate grad with incoming_grad
    for (size_t i = 0; i < data_.size(); ++i) {
        grad_->data_[i] += incoming_grad.data_[i];
    }
}

const Tensor& Tensor::grad() const {
    if (!grad_) {
        throw std::runtime_error("Gradient has not been computed. Call backward() first.");
    }

    return *grad_;
}

void Tensor::zero_grad() {
    if (grad_) {
        grad_->fill(0.0f);
    }
}

void Tensor::backward() {
    if (size() != 1) {
        throw std::runtime_error("backward() can only be called on scalar tensors. Use backward(grad_output) for non-scalar tensors.");
    }

    Tensor grad_output({1}, 1.0f);
    backward(grad_output);
}

void Tensor::backward(const Tensor& grad_output) {
    if (!requires_grad_) {
        return;
    }

    // init this tensors gradient with incoming gradient
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(grad_output);
    } else {
        accumulate_grad(grad_output);
    }
    // call backward func to propogate to inputs
    if (backward_fn_) {
        backward_fn_();
    }
}

} // namespace tiny_transformer