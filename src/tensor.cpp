#include "../include/tiny_transformer/tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <unordered_set>

namespace tiny_transformer {

// ==================== Constructors ====================

Tensor::Tensor() : node_(std::make_shared<Node>()) {}

Tensor::Tensor(const std::vector<size_t>& shape) 
    : node_(std::make_shared<Node>(shape)) {}

Tensor::Tensor(const std::vector<size_t>& shape, float fill) 
    : node_(std::make_shared<Node>(shape, fill)) {}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data) 
    : node_(std::make_shared<Node>(shape, data)) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad) 
    : node_(std::make_shared<Node>(shape)) {
    node_->requires_grad = requires_grad;
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad) 
    : node_(std::make_shared<Node>(shape, data)) {
    node_->requires_grad = requires_grad;
}

// ==================== Accessors ====================

float& Tensor::operator[](size_t idx) {
    return node_->data[idx];
}

const float& Tensor::operator[](size_t idx) const {
    return node_->data[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    if (indices.size() != node_->shape.size()) {
        throw std::out_of_range("Index dimensions don't match tensor dimensions");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= node_->shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    return node_->data[node_->compute_offset(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != node_->shape.size()) {
        throw std::out_of_range("Index dimensions don't match tensor dimensions");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= node_->shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    return node_->data[node_->compute_offset(indices)];
}

void Tensor::check_shape_compatibility(const Tensor& other) const {
    if (node_->shape != other.node_->shape) {
        throw std::invalid_argument("Shape mismatch");
    }
}

// ==================== Autograd ====================

Tensor Tensor::grad() const {
    if (!node_->grad) {
        throw std::runtime_error("Gradient has not been computed. Call backward() first.");
    }
    return Tensor(node_->grad);
}

void Tensor::zero_grad() {
    if (node_->grad) {
        std::fill(node_->grad->data.begin(), node_->grad->data.end(), 0.0f);
    }
}

// Helper: Build topological order of nodes (reverse DFS post-order)
static void build_topo(std::shared_ptr<Node> node, 
                       std::unordered_set<Node*>& visited,
                       std::vector<std::shared_ptr<Node>>& topo) {
    if (!node || visited.count(node.get())) return;
    visited.insert(node.get());
    
    for (auto& parent : node->parents) {
        build_topo(parent, visited, topo);
    }
    topo.push_back(node);
}

void Tensor::backward() {
    if (size() != 1) {
        throw std::runtime_error("backward() requires scalar tensor. Use backward(grad_output) for non-scalar.");
    }
    
    // Build topological order
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> topo;
    build_topo(node_, visited, topo);
    
    // Initialize gradient of output node to 1.0
    node_->grad = std::make_shared<Node>(node_->shape, 1.0f);
    
    // Process nodes in reverse topological order
    // Each node's backward_fn accumulates gradients into its parents
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto& node = *it;
        if (node->backward_fn && node->grad) {
            node->backward_fn();
        }
    }
}

void Tensor::backward(const Tensor& grad_output) {
    // This version is for accumulating gradients (called by backward_fn)
    if (!node_->requires_grad) {
        return;
    }
    
    // Initialize or accumulate gradient (but DON'T call backward_fn)
    if (!node_->grad) {
        node_->grad = std::make_shared<Node>(node_->shape, grad_output.data());
    } else {
        for (size_t i = 0; i < node_->grad->data.size(); ++i) {
            node_->grad->data[i] += grad_output[i];
        }
    }
    // Note: backward_fn is NOT called here anymore - it's called by the topo sort loop
}

// ==================== Operations ====================

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t total_size = 1;
    for (size_t dim : new_shape) {
        total_size *= dim;
    }
    if (total_size != node_->data.size()) {
        throw std::invalid_argument("New shape must have same total size");
    }
    return Tensor(new_shape, node_->data);
}

Tensor Tensor::transpose() const {
    if (node_->shape.size() != 2) {
        throw std::invalid_argument("transpose requires 2D tensor");
    }
    
    size_t rows = node_->shape[0];
    size_t cols = node_->shape[1];
    
    Tensor result({cols, rows});
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at({j, i}) = this->at({i, j});
        }
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    check_shape_compatibility(other);
    
    // Create result with new node
    auto result_node = std::make_shared<Node>(node_->shape);
    for (size_t i = 0; i < node_->data.size(); ++i) {
        result_node->data[i] = node_->data[i] + other.node_->data[i];
    }
    
    // Setup autograd
    if (node_->requires_grad || other.node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_, other.node_};
        
        // Capture the nodes directly - they're shared_ptrs, so graph is preserved
        auto left_node = node_;
        auto right_node = other.node_;
        
        result_node->backward_fn = [result_node, left_node, right_node]() {
            if (!result_node->grad) return;
            
            Tensor grad(result_node->grad);
            
            // d(a+b)/da = 1, d(a+b)/db = 1
            if (left_node->requires_grad) {
                Tensor left(left_node);
                left.backward(grad);
            }
            if (right_node->requires_grad) {
                Tensor right(right_node);
                right.backward(grad);
            }
        };
    }
    
    return Tensor(result_node);
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_shape_compatibility(other);
    
    auto result_node = std::make_shared<Node>(node_->shape);
    for (size_t i = 0; i < node_->data.size(); ++i) {
        result_node->data[i] = node_->data[i] - other.node_->data[i];
    }
    
    if (node_->requires_grad || other.node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_, other.node_};
        
        auto left_node = node_;
        auto right_node = other.node_;
        
        result_node->backward_fn = [result_node, left_node, right_node]() {
            if (!result_node->grad) return;
            
            Tensor grad(result_node->grad);
            
            // d(a-b)/da = 1, d(a-b)/db = -1
            if (left_node->requires_grad) {
                Tensor left(left_node);
                left.backward(grad);
            }
            if (right_node->requires_grad) {
                Tensor right(right_node);
                // Negate gradient
                Tensor neg_grad(right_node->shape);
                for (size_t i = 0; i < neg_grad.size(); ++i) {
                    neg_grad[i] = -grad[i];
                }
                right.backward(neg_grad);
            }
        };
    }
    
    return Tensor(result_node);
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatibility(other);
    
    auto result_node = std::make_shared<Node>(node_->shape);
    for (size_t i = 0; i < node_->data.size(); ++i) {
        result_node->data[i] = node_->data[i] * other.node_->data[i];
    }

    if (node_->requires_grad || other.node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_, other.node_};
        
        auto left_node = node_;
        auto right_node = other.node_;
        
        result_node->backward_fn = [result_node, left_node, right_node]() {
            if (!result_node->grad) return;
            
            // d(a*b)/da = b, d(a*b)/db = a
            if (left_node->requires_grad) {
                Tensor grad_left(left_node->shape);
                for (size_t i = 0; i < grad_left.size(); ++i) {
                    grad_left[i] = result_node->grad->data[i] * right_node->data[i];
                }
                Tensor left(left_node);
                left.backward(grad_left);
            }
            if (right_node->requires_grad) {
                Tensor grad_right(right_node->shape);
                for (size_t i = 0; i < grad_right.size(); ++i) {
                    grad_right[i] = result_node->grad->data[i] * left_node->data[i];
                }
                Tensor right(right_node);
                right.backward(grad_right);
            }
        };
    }
    
    return Tensor(result_node);
}

Tensor Tensor::operator*(float scalar) const {
    auto result_node = std::make_shared<Node>(node_->shape);
    for (size_t i = 0; i < node_->data.size(); ++i) {
        result_node->data[i] = node_->data[i] * scalar;
    }
    

    if (node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_};
        
        auto input_node = node_;
        
        result_node->backward_fn = [result_node, input_node, scalar]() {
            if (!result_node->grad) return;
            
            // d(a*c)/da = c
            Tensor grad_input(input_node->shape);
            for (size_t i = 0; i < grad_input.size(); ++i) {
                grad_input[i] = result_node->grad->data[i] * scalar;
            }
            Tensor input(input_node);
            input.backward(grad_input);
        };
    }
    
    return Tensor(result_node);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (node_->shape.size() != 2 || other.node_->shape.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    if (node_->shape[1] != other.node_->shape[0]) {
        throw std::invalid_argument("Inner dimensions must match for matmul");
    }
    
    size_t m = node_->shape[0];
    size_t k = node_->shape[1];
    size_t n = other.node_->shape[1];
    
    auto result_node = std::make_shared<Node>(std::vector<size_t>{m, n});
    
    // C[i,j] = sum_p A[i,p] * B[p,j]
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += node_->data[i * k + p] * other.node_->data[p * n + j];
            }
            result_node->data[i * n + j] = sum;
        }
    }
    
    if (node_->requires_grad || other.node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_, other.node_};
        
        auto left_node = node_;
        auto right_node = other.node_;
        
        result_node->backward_fn = [result_node, left_node, right_node, m, k, n]() {
            if (!result_node->grad) return;
            
            // dL/dA = dL/dC @ B^T
            if (left_node->requires_grad) {
                Tensor grad_left({m, k});
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < k; ++j) {
                        float sum = 0.0f;
                        for (size_t p = 0; p < n; ++p) {
                            sum += result_node->grad->data[i * n + p] * right_node->data[j * n + p];
                        }
                        grad_left[i * k + j] = sum;
                    }
                }
                Tensor left(left_node);
                left.backward(grad_left);
            }
            
            // dL/dB = A^T @ dL/dC
            if (right_node->requires_grad) {
                Tensor grad_right({k, n});
                for (size_t i = 0; i < k; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (size_t p = 0; p < m; ++p) {
                            sum += left_node->data[p * k + i] * result_node->grad->data[p * n + j];
                        }
                        grad_right[i * n + j] = sum;
                    }
                }
                Tensor right(right_node);
                right.backward(grad_right);
            }
        };
    }
    
    return Tensor(result_node);
}

Tensor Tensor::sum() const {
    float sum_val = 0.0f;
    for (float val : node_->data) {
        sum_val += val;
    }
    
    auto result_node = std::make_shared<Node>(std::vector<size_t>{1}, sum_val);
    
    if (node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_};
        
        auto input_node = node_;
        
        result_node->backward_fn = [result_node, input_node]() {
            if (!result_node->grad) return;
            
            // d(sum)/dx_i = 1 for all i
            // Gradient broadcasts to all elements
            Tensor grad_input(input_node->shape, result_node->grad->data[0]);
            Tensor input(input_node);
            input.backward(grad_input);
        };
    }
    
    return Tensor(result_node);
}

Tensor Tensor::mean() const {
    float mean_val = 0.0f;
    for (float val : node_->data) {
        mean_val += val;
    }
    mean_val /= static_cast<float>(node_->data.size());
    
    auto result_node = std::make_shared<Node>(std::vector<size_t>{1}, mean_val);
    
    if (node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_};
        
        auto input_node = node_;
        size_t n = node_->data.size();
        
        result_node->backward_fn = [result_node, input_node, n]() {
            if (!result_node->grad) return;
            
            // d(mean)/dx_i = 1/n for all i
            float grad_val = result_node->grad->data[0] / static_cast<float>(n);
            Tensor grad_input(input_node->shape, grad_val);
            Tensor input(input_node);
            input.backward(grad_input);
        };
    }
    
    return Tensor(result_node);
}

// ==================== Utilities ====================

void Tensor::fill(float value) {
    std::fill(node_->data.begin(), node_->data.end(), value);
}

void Tensor::print() const {
    if (node_->shape.empty()) {
        std::cout << "Scalar: " << node_->data[0] << std::endl;
        return;
    }
    
    if (node_->shape.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < node_->shape[0]; ++i) {
            std::cout << node_->data[i];
            if (i < node_->shape[0] - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        return;
    }
    
    if (node_->shape.size() == 2) {
        std::cout << "[" << std::endl;
        for (size_t i = 0; i < node_->shape[0]; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < node_->shape[1]; ++j) {
                std::cout << this->at({i, j});
                if (j < node_->shape[1] - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i < node_->shape[0] - 1) std::cout << ",";
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
        return;
    }
    
    std::cout << "Shape: [";
    for (size_t i = 0; i < node_->shape.size(); ++i) {
        std::cout << node_->shape[i];
        if (i < node_->shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], Data: [";
    for (size_t i = 0; i < std::min(node_->data.size(), size_t(10)); ++i) {
        std::cout << node_->data[i];
        if (i < node_->data.size() - 1) std::cout << ", ";
    }
    if (node_->data.size() > 10) std::cout << "...";
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
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dist(gen);
    }
    
    return Tensor(shape, data);
}

} // namespace tiny_transformer