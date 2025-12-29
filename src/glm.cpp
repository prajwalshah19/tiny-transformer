#include "../include/tiny_transformer/glm.hpp"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace tiny_transformer {

GLM::GLM(GLMFamily family, GLMSolver solver, float learning_rate, float tol)
    : family_(family), solver_(solver), lr_(learning_rate), tol_(tol) {}

GLMResult GLM::fit(const Tensor& X, const Tensor& y, std::optional<size_t> max_iter) {
    size_t n_features = X.shape()[1];
    Tensor init_w = Tensor::randn({n_features, 1});
    for (size_t i = 0; i < init_w.size(); ++i) {
        init_w[i] *= 0.01f;
    }
    result_ = fit_impl(X, y, init_w, 0.0f, max_iter);
    fitted_ = true;
    return result_;
}

GLMResult GLM::refit(const Tensor& X, const Tensor& y, std::optional<size_t> max_iter) {
    if (!fitted_) {
        return fit(X, y, max_iter);
    }
    result_ = fit_impl(X, y, result_.weights, result_.bias, max_iter);
    return result_;
}

GLMResult GLM::fit_impl(const Tensor& X, const Tensor& y, const Tensor& init_weights, float init_bias, std::optional<size_t> max_iter) {
    throw std::logic_error("GLM::fit_impl not yet implemented");
}

Tensor GLM::predict(const Tensor& X) const {
    if (!fitted_) throw std::runtime_error("Model not fitted. Call fit() first.");
    return predict(X, result_);
}

float GLM::predict_one(const Tensor& x) const {
    if (!fitted_) throw std::runtime_error("Model not fitted. Call fit() first.");
    return predict_one(x, result_);
}

Tensor GLM::predict(const Tensor& X, const GLMResult& result) const {
    throw std::logic_error("GLM::predict not yet implemented");
}

float GLM::predict_one(const Tensor& x, const GLMResult& result) const {
    throw std::logic_error("GLM::predict_one not yet implemented");
}

const GLMResult& GLM::result() const {
    if (!fitted_) throw std::runtime_error("No result available. Call fit() first.");
    return result_;
}

float GLM::compute_loss(const Tensor& y_pred, const Tensor& y) const {
    throw std::logic_error("GLM::compute_loss not yet implemented");
}

float GLM::compute_r_squared(const Tensor& y_pred, const Tensor& y) const {
    throw std::logic_error("GLM::compute_r_squared not yet implemented");
}

float GLM::grad_norm(const Tensor& w_grad, float b_grad) const {
    throw std::logic_error("GLM::grad_norm not yet implemented");
}

} // namespace tiny_transformer
