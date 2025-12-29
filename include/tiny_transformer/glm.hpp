#pragma once
#include "tensor.hpp"
#include <optional>
#include <string>
#include <vector>

namespace tiny_transformer {

enum class GLMFamily {
    Gaussian,   // Linear regression
    Bernoulli,  // Logistic regression
    Multinomial // Softmax regression
};

enum class GLMSolver {
    GradientDescent,
    Newton
};

struct GLMResult {
    Tensor weights;
    float bias;
    float loss;
    float r2;
    size_t n_iter;
    bool converged;
};

class GLM {
public:
    GLM(GLMFamily family, GLMSolver solver = GLMSolver::GradientDescent, float learning_rate = 1e-2, float tol = 1e-6);

    GLMResult fit(const Tensor& X, const Tensor& y, std::optional<size_t> max_iter = std::nullopt);
    GLMResult refit(const Tensor& X, const Tensor& y, std::optional<size_t> max_iter = std::nullopt);
    Tensor predict(const Tensor& X) const;
    float predict_one(const Tensor& x) const;
    const GLMResult& result() const;

private:
    GLMResult fit_impl(const Tensor& X, const Tensor& y, const Tensor& init_weights, float init_bias, std::optional<size_t> max_iter);
    Tensor predict(const Tensor& X, const GLMResult& result) const;
    float predict_one(const Tensor& x, const GLMResult& result) const;
    float compute_loss(const Tensor& y_pred, const Tensor& y) const;
    float compute_r_squared(const Tensor& y_pred, const Tensor& y) const;
    float grad_norm(const Tensor& w_grad, float b_grad) const;

    GLMFamily family_;
    GLMSolver solver_;
    float lr_;
    float tol_;
    bool fitted_ = false;
    GLMResult result_;
};

} // namespace tiny_transformer
