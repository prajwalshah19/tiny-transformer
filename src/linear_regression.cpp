#include "../include/tiny_transformer/linear_regression.hpp"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace tiny_transformer {

LinearRegression::LinearRegression(float learning_rate, float tol)
    : lr_(learning_rate), tol_(tol) {}

OLSResult LinearRegression::fit(const Tensor& X, const Tensor& y, 
                                 std::optional<size_t> max_iter) {
    size_t n_features = X.shape()[1];
    
    // fresh weights, refit() updates with existing result
    Tensor init_w = Tensor::randn({n_features, 1});
    for (size_t i = 0; i < init_w.size(); ++i) {
        init_w[i] *= 0.01f; // regular normal dist values are TOO large, must scale down
    }
    
    result_ = fit_impl(X, y, init_w, 0.0f, max_iter);
    fitted_ = true;
    return result_;
}

OLSResult LinearRegression::refit(const Tensor& X, const Tensor& y,
                                   std::optional<size_t> max_iter) {
    if (!fitted_) {
        return fit(X, y, max_iter);  // fallback to fresh fit
    }
    
    // Warm start from previous weights
    result_ = fit_impl(X, y, result_.weights, result_.bias, max_iter);
    return result_;
}

// private wrapper behind fit() and refit()
OLSResult LinearRegression::fit_impl(const Tensor& X, const Tensor& y,
                                      const Tensor& init_weights, float init_bias,
                                      std::optional<size_t> max_iter) {
    // Validate inputs
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be 2D (n_samples, n_features)");
    }
    if (y.ndim() != 2 || y.shape()[1] != 1) {
        throw std::invalid_argument("y must be 2D (n_samples, 1)");
    }
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have same number of samples");
    }
    
    size_t n_features = X.shape()[1];
    
    // Create autograd-enabled weight tensors
    Tensor W({n_features, 1}, true);
    Tensor b({1}, std::vector<float>{init_bias}, true);
    for (size_t i = 0; i < n_features; ++i) {
        W[i] = init_weights[i];
    }
    
    size_t max_iterations = max_iter.value_or(std::numeric_limits<size_t>::max());
    size_t iter = 0;
    bool converged = false;
    
    while (iter < max_iterations) {
        // Forward: y_pred = X @ W + b
        Tensor xw = X.matmul(W);
        Tensor y_pred = xw.add_broadcast(b);
        
        // MSE Loss
        Tensor diff = y_pred - y;
        Tensor sq_diff = diff * diff;
        Tensor loss = sq_diff.mean();
        
        // Zero gradients & backward
        W.zero_grad();
        b.zero_grad();
        loss.backward();
        
        // Get gradients
        Tensor w_grad = W.grad();
        Tensor b_grad = b.grad();
        
        // Check convergence
        float gn = grad_norm(w_grad, b_grad[0]);
        if (gn < tol_) {
            converged = true;
            break;
        }
        
        // SGD update
        for (size_t i = 0; i < W.size(); ++i) {
            W[i] -= lr_ * w_grad[i];
        }
        b[0] -= lr_ * b_grad[0];
        
        ++iter;
    }
    
    // Build result
    Tensor final_weights({n_features, 1});
    for (size_t i = 0; i < n_features; ++i) {
        final_weights[i] = W[i];
    }
    float final_bias = b[0];
    
    // Compute metrics
    Tensor y_pred = predict(X, OLSResult{final_weights, final_bias, 0, 0, 0, false});
    float mse = compute_mse(y_pred, y);
    float r2 = compute_r_squared(y_pred, y);
    
    return OLSResult{final_weights, final_bias, mse, r2, iter, converged};
}

// ===== Stateful predict =====
Tensor LinearRegression::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model not fitted. Call fit() first.");
    }
    return predict(X, result_);
}

float LinearRegression::predict_one(const Tensor& x) const {
    if (!fitted_) {
        throw std::runtime_error("Model not fitted. Call fit() first.");
    }
    return predict_one(x, result_);
}

// ===== Static predict =====
Tensor LinearRegression::predict(const Tensor& X, const OLSResult& result) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be 2D");
    }
    
    Tensor xw = X.matmul(result.weights);
    Tensor out({xw.shape()[0], 1});
    for (size_t i = 0; i < xw.size(); ++i) {
        out[i] = xw[i] + result.bias;
    }
    return out;
}

float LinearRegression::predict_one(const Tensor& x, const OLSResult& result) {
    if (x.ndim() != 1) {
        throw std::invalid_argument("x must be 1D (n_features,)");
    }
    
    float out = result.bias;
    for (size_t i = 0; i < x.size(); ++i) {
        out += x[i] * result.weights[i];
    }
    return out;
}

const OLSResult& LinearRegression::result() const {
    if (!fitted_) {
        throw std::runtime_error("No result available. Call fit() first.");
    }
    return result_;
}

// ===== Helpers =====
float LinearRegression::compute_mse(const Tensor& y_pred, const Tensor& y) {
    float sum = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        float diff = y_pred[i] - y[i];
        sum += diff * diff;
    }
    return sum / static_cast<float>(y.size());
}

float LinearRegression::compute_r_squared(const Tensor& y_pred, const Tensor& y) {
    float y_mean = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        y_mean += y[i];
    }
    y_mean /= static_cast<float>(y.size());
    
    float ss_res = 0.0f, ss_tot = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        float diff_pred = y[i] - y_pred[i];
        float diff_mean = y[i] - y_mean;
        ss_res += diff_pred * diff_pred;
        ss_tot += diff_mean * diff_mean;
    }
    
    return (ss_tot == 0.0f) ? 1.0f : 1.0f - (ss_res / ss_tot);
}

float LinearRegression::grad_norm(const Tensor& w_grad, float b_grad) const {
    float norm_sq = b_grad * b_grad;
    for (size_t i = 0; i < w_grad.size(); ++i) {
        norm_sq += w_grad[i] * w_grad[i];
    }
    return std::sqrt(norm_sq);
}

} // namespace tiny_transformer