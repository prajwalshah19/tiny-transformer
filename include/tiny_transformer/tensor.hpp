#ifndef TINY_TRANSFORMER_TENSOR_HPP
#define TINY_TRANSFORMER_TENSOR_HPP

#include <vector>
#include <memory>
#include <functional>

namespace tiny_transformer {

/**
 * @brief Internal node that holds actual tensor data and autograd information.
 * This is the "graph node" that persists across Tensor copies.
 * Makes it easier for intermediary nodes in autograd calculation to persist safely
 */
struct Node {
    // Data
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    
    // Autograd
    std::shared_ptr<Node> grad;
    bool requires_grad = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_fn;
    
    // Constructors
    Node() : data({0.0f}), shape({}), strides({}) {}
    
    Node(const std::vector<size_t>& shape_) : shape(shape_) {
        compute_strides();
        size_t total = 1;
        for (size_t dim : shape) total *= dim;
        data.resize(total, 0.0f);
    }
    
    Node(const std::vector<size_t>& shape_, float fill) : shape(shape_) {
        compute_strides();
        size_t total = 1;
        for (size_t dim : shape) total *= dim;
        data.resize(total, fill);
    }
    
    Node(const std::vector<size_t>& shape_, const std::vector<float>& data_) 
        : data(data_), shape(shape_) {
        compute_strides();
    }
    
    void compute_strides() {
        if (shape.empty()) {
            strides.clear();
            return;
        }
        strides.resize(shape.size());
        size_t val = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = val;
            val *= shape[i];
        }
    }
    
    size_t compute_offset(const std::vector<size_t>& indices) const {
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            offset += strides[i] * indices[i];
        }
        return offset;
    }
};

/**
 * @brief Lightweight tensor handle that wraps a shared Node.
 * Copying a Tensor shares the underlying Node, preserving the computation graph.
 */
class Tensor {
private:
    std::shared_ptr<Node> node_;
    
    void check_shape_compatibility(const Tensor& other) const;

public:
    // ==================== Constructors ====================
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float fill);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
    Tensor(const std::vector<size_t>& shape, bool requires_grad);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad);
    
    // Construct from existing node (internal use)
    explicit Tensor(std::shared_ptr<Node> node) : node_(node) {}

    // ==================== Accessors ====================
    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    
    size_t size() const { return node_->data.size(); }
    size_t ndim() const { return node_->shape.size(); }
    const std::vector<size_t>& shape() const { return node_->shape; }
    const std::vector<float>& data() const { return node_->data; }
    std::vector<float>& data() { return node_->data; }
    
    // Access underlying node (for advanced usage)
    std::shared_ptr<Node> node() const { return node_; }

    // ==================== Autograd ====================
    bool requires_grad() const { return node_->requires_grad; }
    void set_requires_grad(bool req) { node_->requires_grad = req; }
    
    Tensor grad() const;
    bool has_grad() const { return node_->grad != nullptr; }
    void zero_grad();
    void backward();
    void backward(const Tensor& grad_output);

    // ==================== Operations ====================
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor transpose() const;
    
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    
    Tensor sum() const;
    Tensor mean() const;

    // ==================== Utilities ====================
    void fill(float value);
    void print() const;
    
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor randn(const std::vector<size_t>& shape);
};

} // namespace tiny_transformer

#endif