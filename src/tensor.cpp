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
    
    // Build topological order, this eliminates branching off and reaccumulating because of repeats
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
    
    // Initialize or accumulate gradient - first phase
    if (!node_->grad) {
        node_->grad = std::make_shared<Node>(node_->shape, grad_output.data());
    } else {
        for (size_t i = 0; i < node_->grad->data.size(); ++i) {
            node_->grad->data[i] += grad_output[i];
        }
    }
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

Tensor Tensor::add_broadcast(const Tensor& other) const {
    // Handle simple case: other is scalar-like {1} or same shape
    if (other.shape() == node_->shape) {
        return *this + other;
    }
    
    // Broadcast other across this tensor
    // Common case: this is (n, m), other is (m,) or (1, m) or (1,)
    const auto& a_shape = node_->shape;
    const auto& b_shape = other.shape();
    
    // Compute output shape (max of aligned dimensions)
    size_t ndim_a = a_shape.size();
    size_t ndim_b = b_shape.size();
    size_t ndim_out = std::max(ndim_a, ndim_b);
    
    // Pad shapes from the left with 1s for alignment
    std::vector<size_t> a_padded(ndim_out, 1);
    std::vector<size_t> b_padded(ndim_out, 1);
    
    for (size_t i = 0; i < ndim_a; ++i) {
        a_padded[ndim_out - ndim_a + i] = a_shape[i];
    }
    for (size_t i = 0; i < ndim_b; ++i) {
        b_padded[ndim_out - ndim_b + i] = b_shape[i];
    }
    
    // Compute output shape and validate compatibility
    std::vector<size_t> out_shape(ndim_out);
    for (size_t i = 0; i < ndim_out; ++i) {
        if (a_padded[i] != b_padded[i] && a_padded[i] != 1 && b_padded[i] != 1) {
            throw std::invalid_argument("Shapes not broadcastable");
        }
        out_shape[i] = std::max(a_padded[i], b_padded[i]);
    }
    
    // Compute strides for broadcasting (0 stride means repeat)
    auto compute_broadcast_strides = [&](const std::vector<size_t>& padded_shape) {
        std::vector<size_t> strides(ndim_out);
        size_t stride = 1;
        for (int i = ndim_out - 1; i >= 0; --i) {
            strides[i] = (padded_shape[i] == 1) ? 0 : stride;
            stride *= padded_shape[i];
        }
        return strides;
    };
    
    auto a_strides = compute_broadcast_strides(a_padded);
    auto b_strides = compute_broadcast_strides(b_padded);
    
    // Create result
    auto result_node = std::make_shared<Node>(out_shape);
    
    // Compute total size and iterate
    size_t total = 1;
    for (size_t d : out_shape) total *= d;
    
    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        // Convert flat index to multi-dimensional index
        std::vector<size_t> idx(ndim_out);
        size_t tmp = flat_idx;
        for (int i = ndim_out - 1; i >= 0; --i) {
            idx[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }
        
        // Compute source indices using broadcast strides
        size_t a_idx = 0, b_idx = 0;
        for (size_t i = 0; i < ndim_out; ++i) {
            a_idx += idx[i] * a_strides[i];
            b_idx += idx[i] * b_strides[i];
        }
        
        result_node->data[flat_idx] = node_->data[a_idx] + other.node_->data[b_idx];
    }
    
    // Setup autograd
    if (node_->requires_grad || other.node_->requires_grad) {
        result_node->requires_grad = true;
        result_node->parents = {node_, other.node_};
        
        auto left_node = node_;
        auto right_node = other.node_;
        auto left_shape = a_shape;
        auto right_shape = b_shape;
        
        result_node->backward_fn = [result_node, left_node, right_node, 
                                     left_shape, right_shape, out_shape, ndim_out]() {
            if (!result_node->grad) return;
            
            // For the left operand: sum over broadcasted dimensions
            if (left_node->requires_grad) {
                Tensor grad_left(left_shape, 0.0f);
                
                size_t total = 1;
                for (size_t d : out_shape) total *= d;
                
                // Pad left shape
                size_t ndim_l = left_shape.size();
                std::vector<size_t> l_padded(ndim_out, 1);
                for (size_t i = 0; i < ndim_l; ++i) {
                    l_padded[ndim_out - ndim_l + i] = left_shape[i];
                }
                
                for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
                    std::vector<size_t> idx(ndim_out);
                    size_t tmp = flat_idx;
                    for (int i = ndim_out - 1; i >= 0; --i) {
                        idx[i] = tmp % out_shape[i];
                        tmp /= out_shape[i];
                    }
                    
                    // Map to left index (collapse broadcasted dims)
                    std::vector<size_t> left_idx(ndim_l);
                    for (size_t i = 0; i < ndim_l; ++i) {
                        size_t out_i = ndim_out - ndim_l + i;
                        left_idx[i] = (l_padded[out_i] == 1) ? 0 : idx[out_i];
                    }
                    
                    size_t left_flat = 0;
                    size_t stride = 1;
                    for (int i = ndim_l - 1; i >= 0; --i) {
                        left_flat += left_idx[i] * stride;
                        stride *= left_shape[i];
                    }
                    
                    grad_left[left_flat] += result_node->grad->data[flat_idx];
                }
                
                Tensor left(left_node);
                left.backward(grad_left);
            }
            
            // For the right operand: sum over broadcasted dimensions  
            if (right_node->requires_grad) {
                Tensor grad_right(right_shape, 0.0f);
                
                size_t total = 1;
                for (size_t d : out_shape) total *= d;
                
                size_t ndim_r = right_shape.size();
                std::vector<size_t> r_padded(ndim_out, 1);
                for (size_t i = 0; i < ndim_r; ++i) {
                    r_padded[ndim_out - ndim_r + i] = right_shape[i];
                }
                
                for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
                    std::vector<size_t> idx(ndim_out);
                    size_t tmp = flat_idx;
                    for (int i = ndim_out - 1; i >= 0; --i) {
                        idx[i] = tmp % out_shape[i];
                        tmp /= out_shape[i];
                    }
                    
                    std::vector<size_t> right_idx(ndim_r);
                    for (size_t i = 0; i < ndim_r; ++i) {
                        size_t out_i = ndim_out - ndim_r + i;
                        right_idx[i] = (r_padded[out_i] == 1) ? 0 : idx[out_i];
                    }
                    
                    size_t right_flat = 0;
                    size_t stride = 1;
                    for (int i = ndim_r - 1; i >= 0; --i) {
                        right_flat += right_idx[i] * stride;
                        stride *= right_shape[i];
                    }
                    
                    grad_right[right_flat] += result_node->grad->data[flat_idx];
                }
                
                Tensor right(right_node);
                right.backward(grad_right);
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

// ==================== GLM Activation FUncs ====================

Tensor Tensor::relu() const {
    auto out_node = std::make_shared<Node>(node_->shape);
    size_t n = node_->data.size();
    out_node->data.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float v = node_->data[i];
        out_node->data[i] = v > 0.0f ? v : 0.0f;
    }

    out_node->requires_grad = node_->requires_grad;
    if (node_->requires_grad) {
        out_node->parents = { node_ };
        std::weak_ptr<Node> out_wp = out_node;
        std::weak_ptr<Node> in_wp = node_;
        out_node->backward_fn = [out_wp, in_wp]() {
            auto out_sp = out_wp.lock();
            auto in_sp = in_wp.lock();
            if (!out_sp || !in_sp) return;
            if (!out_sp->grad) return;
            if (!in_sp->grad) in_sp->grad = std::make_shared<Node>(in_sp->shape, 0.0f);
            size_t m = in_sp->data.size();
            for (size_t j = 0; j < m; ++j) {
                float vin = in_sp->data[j];
                float grad_out = out_sp->grad->data[j];
                float deriv = vin > 0.0f ? 1.0f : 0.0f; // subgradient at 0 chosen as 0
                in_sp->grad->data[j] += grad_out * deriv;
            }
        };
    }

    return Tensor(out_node);
}

Tensor Tensor::sigmoid() const {
    auto out_node = std::make_shared<Node>(node_->shape);
    size_t n = node_->data.size();
    out_node->data.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out_node->data[i] = 1.0f / (1.0f + std::exp(-node_->data[i]));
    }

    out_node->requires_grad = node_->requires_grad;
    if (node_->requires_grad) {
        // keep parent reference for backward
        out_node->parents = { node_ };
        // weak_ptrs to avoid cycles
        std::weak_ptr<Node> out_wp = out_node;
        std::weak_ptr<Node> in_wp = node_;
        out_node->backward_fn = [out_wp, in_wp]() {
            auto out_sp = out_wp.lock();
            auto in_sp = in_wp.lock();
            if (!out_sp || !in_sp) return;
            if (!out_sp->grad) return; // nothing to backprop
            if (!in_sp->grad) in_sp->grad = std::make_shared<Node>(in_sp->shape, 0.0f);
            size_t m = in_sp->data.size();
            for (size_t j = 0; j < m; ++j) {
                float s = out_sp->data[j]; // sigmoid(x)
                float grad_out = out_sp->grad->data[j];
                // derivative: s * (1 - s)
                in_sp->grad->data[j] += grad_out * s * (1.0f - s);
            }
        };
    }

    return Tensor(out_node);
}
Tensor Tensor::softmax() const {
    // along the last axis (numerically stable)
    // what does this mean? - compute the softmax independently across different batches, independently normalize features
    // ex, suppose we have dims for batch, sample, we dont want feature softmax to entangle, ruins IID
    const auto& shape = node_->shape;
    if (shape.empty()) {
        // scalar -> softmax is 1
        return Tensor(std::make_shared<Node>(std::vector<size_t>{1}, std::vector<float>{1.0f}));
    }

    size_t last_dim = shape.back();
    size_t total = node_->data.size();
    size_t blocks = total / last_dim; // number of independent vectors

    auto out_node = std::make_shared<Node>(shape);
    out_node->data.resize(total);

    // compute softmax per block
    for (size_t b = 0; b < blocks; ++b) {
        size_t offset = b * last_dim;
        // find max for numerical stability
        float m = node_->data[offset];
        for (size_t j = 1; j < last_dim; ++j) {
            float v = node_->data[offset + j];
            if (v > m) m = v;
        }
        // exponentiate and sum
        float sum = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            float e = std::exp(node_->data[offset + j] - m);
            out_node->data[offset + j] = e;
            sum += e;
        }
        // normalize
        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < last_dim; ++j) {
            out_node->data[offset + j] *= inv_sum;
        }
    }

    out_node->requires_grad = node_->requires_grad;
    if (node_->requires_grad) {
        out_node->parents = { node_ };
        std::weak_ptr<Node> out_wp = out_node;
        std::weak_ptr<Node> in_wp = node_;
        size_t ld = last_dim;
        size_t blks = blocks;
        out_node->backward_fn = [out_wp, in_wp, ld, blks]() {
            auto out_sp = out_wp.lock();
            auto in_sp = in_wp.lock();
            if (!out_sp || !in_sp) return;
            if (!out_sp->grad) return;
            if (!in_sp->grad) in_sp->grad = std::make_shared<Node>(in_sp->shape, 0.0f);

            // For each block compute: grad_in = softmax * (grad_out - dot(grad_out, softmax))
            for (size_t b = 0; b < blks; ++b) {
                size_t off = b * ld;
                // dot = sum_j grad_out_j * s_j
                float dot = 0.0f;
                for (size_t j = 0; j < ld; ++j) {
                    dot += out_sp->grad->data[off + j] * out_sp->data[off + j];
                }
                for (size_t i = 0; i < ld; ++i) {
                    float s_i = out_sp->data[off + i];
                    float g_out = out_sp->grad->data[off + i];
                    float g_in = s_i * (g_out - dot);
                    in_sp->grad->data[off + i] += g_in;
                }
            }
        };
    }

    return Tensor(out_node);
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