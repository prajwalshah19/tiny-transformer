#ifndef TINY_TRANSFORMER_TENSOR_HPP
#define TINY_TRANSFORMER_TENSOR_HPP

#include <vector>
#include <memory>
#include <initializer_list>

namespace tiny_transformer {

class Tensor {
    // fields
    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

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

    // access
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return data_.size(); }

    // data access
    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }

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