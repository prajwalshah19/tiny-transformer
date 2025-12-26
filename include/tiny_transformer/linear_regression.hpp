#pragma once

#include "tensor.hpp"
#include <optional>

namespace tiny_transformer {

struct OLSResult {
    Tensor weights;       // (n_features, 1) - learned coefficients
    float bias;           // intercept term
    float mse;            // mean squared error on training data
    float r_squared;      // coefficient of determination
    size_t iterations;    // iterations to converge
    bool converged;       // did gradient descent converge?
};

class LinearRegression {
public:
    LinearRegression(float learning_rate = 0.01f, float tol = 1e-6f);
    
    // ===== Stateful API =====
    // Fit and store result internally while returning it for inspection
    OLSResult fit(const Tensor& X, const Tensor& y, 
                  std::optional<size_t> max_iter = 1000);
    
    // Predict using stored weights (must call fit() first)
    Tensor predict(const Tensor& X) const;
    float predict_one(const Tensor& x) const;
    
    // Continue training from current weights, update result_
    OLSResult refit(const Tensor& X, const Tensor& y,
                    std::optional<size_t> max_iter = 1000);
    
    // ====== Static API =====
    // Predict using explicit result (no fit() needed)
    static Tensor predict(const Tensor& X, const OLSResult& result);
    static float predict_one(const Tensor& x, const OLSResult& result);
    
    // ===== Accessors =====
    const OLSResult& result() const;  // get last fit result
    bool is_fitted() const { return fitted_; }
    
private:
    float lr_;
    float tol_;
    OLSResult result_;     // most recent fit
    bool fitted_ = false;
    
    // Internal fit implementation
    OLSResult fit_impl(const Tensor& X, const Tensor& y,
                       const Tensor& init_weights, float init_bias,
                       std::optional<size_t> max_iter);
    
    // Helpers
    static float compute_mse(const Tensor& y_pred, const Tensor& y);
    static float compute_r_squared(const Tensor& y_pred, const Tensor& y);
    float grad_norm(const Tensor& w_grad, float b_grad) const;
};

} // namespace tiny_transformer