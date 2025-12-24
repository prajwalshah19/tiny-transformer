#ifndef TINY_TRANSFORMER_TENSOR_HPP
#define TINY_TRANSFORMER_TENSOR_HPP

#include <vector>
#include <memory>
#include <initializer_list>
#include <functional>

namespace tiny_transformer {

class Tensor : public std::enable_shared_from_this<Tensor> {
    // fields
    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<std::shared_ptr<Tensor>> parents_;

    std::shared_ptr<Tensor> grad_; // gradient of this tensor
    bool requires_grad_ = false;
    std::function<void()> backward_fn_; // function to backpropagate one level

     /**
     * @brief Accumlates gradient
     * 
     * 
     * @return None, modifies grad_ in place
     */
    void accumulate_grad(const Tensor& incoming_grad);

    /**
     * @brief Compute strides to dicate movement in memory to increment
     * strides_[0 ... n] represent every dimension, used to compte offset when
     * accessing certain params.
     * 
     * How does it work? By building the offset backwards.
     * 
     * @return None, modidies strides_ in place
     */
    void compute_strides();

    /**
     * @brief Use the strides and given 
     * 
     * @param indices a multidimensional index
     * @return Offset to access that index in data_
     */
    size_t compute_offset(const std::vector<size_t>& indices) const;

     /**
     * @brief Checks if two tensors are the same shape
     * 
     * @param other another tensor to check
     * @return None, throws error
     */
    void check_shape_compatibility(const Tensor &other) const;
public:
    // constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float fill);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
    Tensor(const std::vector<size_t>& shape, bool requires_grad);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad);

    // access
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return data_.size(); }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    // data access
    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    bool has_grad() const { return grad_ != nullptr; }
    const Tensor& grad() const;

    
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }

    // autograd
    /**
     * @brief For scalar outputs, calls backward(const Tensor&) with 1-filled gradient
     * 
     * @return None, Modifies grad_ in-place
     */
    void backward();

     /**
     * @brief For any output, accumulates gradient w/incoming and then applices backward_fn
     * 
     * @return None, Modifies grad_ in-place
     */
    void backward(const Tensor& grad_output); // non scalar outputs

    /**
     * @brief Resets gradient to avoid reaccumulation
     * 
     * @return None, Modifies grad_ in-place
     */
    void zero_grad();

    // shape ops
    /**
     * @brief Reshapes a tensor w/o changing underlying data
     * 
     * @param new_shape another shape with same total size
     * @return New tensor with different shape
     */
    Tensor reshape(const std::vector<size_t>& new_shape) const;

    /**
     * @brief Performs a transpose (reverse row and col) on a 2d tensor
     * 
     * @return New transposed tensor
     */
    Tensor transpose() const;

    // element wise

    /**
     * @brief Add two same shape Tensors
     * 
     * @return New sum tensor
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Subtract (add negative) two same shape Tensors
     * 
     * @return New difference tensor
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Product of two same shape Tensors
     * 
     * @return New product tensor
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Scalar product of two same shape Tensors
     * 
     * @return New product tensor
     */
    Tensor operator*(float scalar) const;

    // matrix mult
    Tensor matmul(const Tensor& other) const;

    // reduction ops
    Tensor sum() const;
    Tensor mean() const;

    // utils
    void fill(float value);
    void print() const;

    // factory methods
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor randn(const std::vector<size_t>& shape);

}; // Tensor class

} // namespace 

#endif