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
    // X is 2 dims [samples, features]
    const auto x_shape = X.shape();
    if (x_shape.size() < 2) throw std::invalid_argument("X must be 2-D [n_samples, n_features]");
    size_t n_samples = x_shape[0];
    size_t n_features = x_shape[1];

    // y must be formated correctly, theta^T X = y
    // n_samp * n_feat X n_feat * out_dim
    size_t out_dim = 1;
    const auto w_shape = init_weights.shape();
    if (w_shape.size() == 2) out_dim = w_shape[1];
    else if (w_shape.size() == 1) out_dim = 1;
    else throw std::invalid_argument("init_weights must be 1-D or 2-D");
    // y has shape [n_sam, out_dim]

    // ensure y shape
    Tensor y_mat = y;
    const auto y_shape = y.shape();
    if (y_shape.size() == 1 && out_dim == 1) {
        y_mat = y.reshape({n_samples, 1});
    } else if (y_shape.size() == 2) {
        // assume [n_samples, out_dim]
        if (y_shape[0] != n_samples) throw std::invalid_argument("y and X sample mismatch");
        if (y_shape[1] != out_dim && y_shape[1] != 1 && out_dim != 1) {
            // allow broadcasting y with single column, otherwise mismatch
            if (!(y_shape[1] == 1 || out_dim == 1)) throw std::invalid_argument("y and init_weights output dim mismatch");
        }
        if (y_shape[1] != out_dim) {
            // reshape y to [n_samples, out_dim] by broadcasting its single column if needed
            if (y_shape[1] == 1) {
                // leave as is; broadcasting ops below will handle shapes
            } else {
                // keep as provided (may be handled by matmul)
            }
        }
    } else {
        throw std::invalid_argument("Unsupported y shape");
    }

    Tensor w = init_weights;
    float b = init_bias;

    // actual learning process
    size_t max_iters = max_iter.value_or(1000);
    const float eps = 1e-8f;
    bool converged = false;
    size_t iter = 0;

    // update rule
    // eta = X @ w + b
    for (iter = 1; iter <= max_iters; ++iter) {
        Tensor eta = X.matmul(w);
        Tensor b_tensor = Tensor::ones({n_samples, out_dim}) * b;
        eta = eta + b_tensor;

        // mu (mean) = inverse-link func / prediction
        Tensor mu;
        if (family_ == GLMFamily::Gaussian) {
            mu = eta; // identity, shifting a gaussian by mean (def = 0)
        } else if (family_ == GLMFamily::Bernoulli) {
            mu = eta.sigmoid();
        } else if (family_ == GLMFamily::Multinomial) {
            mu = eta.softmax();
        } else {
            throw std::logic_error("Unsupported GLM family");
        }

        Tensor residual = mu - y_mat;
        Tensor w_grad = X.transpose().matmul(residual) * (1.0f / static_cast<float>(n_samples));
        Tensor b_grad_t = residual.mean();
        float b_grad = b_grad_t[0];

         float scale = 1.0f;
        if (solver_ == GLMSolver::Newton) {
            if (family_ == GLMFamily::Gaussian) {
                // dont need to scale gradient
                scale = 1.0f;
            } else if (family_ == GLMFamily::Bernoulli) {
                // scale gradient by avg variance
                float sum_var = 0.0f;
                const auto& mu_data = mu.data();
                for (size_t i = 0; i < mu_data.size(); ++i) sum_var += mu_data[i] * (1.0f - mu_data[i]);
                float avg_var = sum_var / static_cast<float>(mu_data.size());
                scale = 1.0f / (avg_var + eps);
            } else {
                // multinomial
                float sum_var = 0.0f;
                const auto& mu_data = mu.data();
                for (size_t i = 0; i < mu_data.size(); ++i) sum_var += mu_data[i] * (1.0f - mu_data[i]);
                float avg_var = sum_var / static_cast<float>(mu_data.size());
                scale = 1.0f / (avg_var + eps);

            }
        }

        // SGD update rule
        Tensor w_update = w_grad * (lr_ * scale);
        w = w - w_update;
        float b_update = lr_ * scale * b_grad;
        b -= b_update;

        float gnorm = 0.0f;
        {
            const auto& wg = w_grad.data();
            for (float v : wg) gnorm += v * v;
            gnorm += b_grad * b_grad;
            gnorm = std::sqrt(gnorm);
        }

        float param_change_norm = 0.0f;
        {
            const auto& wu = w_update.data();
            for (float v : wu) param_change_norm += v * v;
            param_change_norm += b_update * b_update;
            param_change_norm = std::sqrt(param_change_norm);
        }

        // stop if weights stop changing
        if (gnorm < tol_ || param_change_norm < tol_) {
            converged = true;
            break;
        }
    }

    // final predictions
    Tensor eta_final = X.matmul(w) + (Tensor::ones({n_samples, out_dim}) * b);
    Tensor mu_final;
    if (family_ == GLMFamily::Gaussian) mu_final = eta_final;
    else if (family_ == GLMFamily::Bernoulli) mu_final = eta_final.sigmoid();
    else mu_final = eta_final.softmax();

    float loss = 0.0f;
    const auto& mu_data = mu_final.data();
    const auto& y_data = y_mat.data();
    // loss ofr gaussian is MSE, for bernoulli its binary cross-entropy, mulitnomial is reg cross entropy
    if (family_ == GLMFamily::Gaussian) {
        double sse = 0.0;
        for (size_t i = 0; i < mu_data.size(); ++i) {
            double d = static_cast<double>(y_data[i]) - static_cast<double>(mu_data[i]);
            sse += d * d;
        }
        loss = static_cast<float>(sse / static_cast<double>(mu_data.size()) * 0.5); // 0.5*MSE
    } else if (family_ == GLMFamily::Bernoulli) {
        double acc = 0.0;
        for (size_t i = 0; i < mu_data.size(); ++i) {
            double p = std::min(1.0 - 1e-12, std::max(1e-12, static_cast<double>(mu_data[i])));
            double t = static_cast<double>(y_data[i]);
            acc += -(t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
        }
        loss = static_cast<float>(acc / static_cast<double>(mu_data.size()));
    } else {
        size_t rows = n_samples;
        size_t cols = out_dim;
        double acc = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                size_t idx = r * cols + c;
                double p = std::min(1.0 - 1e-12, std::max(1e-12, static_cast<double>(mu_data[idx])));
                double t = static_cast<double>(y_data[idx]);
                if (t > 0.0) acc += -t * std::log(p);
            }
        }
        loss = static_cast<float>(acc / static_cast<double>(rows));
    }

    // gaussian will have residual
    float r2 = std::numeric_limits<float>::quiet_NaN();
    if (family_ == GLMFamily::Gaussian) {
        double mean_y = 0.0;
        for (float v : y_data) mean_y += v;
        mean_y /= static_cast<double>(y_data.size());
        double sst = 0.0;
        double sse = 0.0;
        for (size_t i = 0; i < y_data.size(); ++i) {
            double dy = static_cast<double>(y_data[i]) - mean_y;
            sst += dy * dy;
            double res = static_cast<double>(y_data[i]) - static_cast<double>(mu_data[i]);
            sse += res * res;
        }
        if (sst > 0.0) r2 = 1.0f - static_cast<float>(sse / sst);
        else r2 = 0.0f;
    }

    // create result object
    GLMResult res;
    res.weights = w;
    res.bias = b;
    res.loss = loss;
    res.r2 = r2;
    res.n_iter = iter;
    res.converged = converged;

    result_ = res;
    return res;

}

Tensor GLM::predict(const Tensor& X, const GLMResult& result) const {
    // X: [n_samples, n_features]
    const auto x_shape = X.shape();
    if (x_shape.size() < 2) throw std::invalid_argument("X must be 2-D [n_samples, n_features]");
    size_t n_samples = x_shape[0];

    const auto& w = result.weights;
    float b = result.bias;

    // determine out_dim from weights
    size_t out_dim = 1;
    const auto w_shape = w.shape();
    if (w_shape.size() == 2) out_dim = w_shape[1];
    else if (w_shape.size() == 1) out_dim = 1;
    else throw std::invalid_argument("result.weights must be 1-D or 2-D");

    Tensor eta = X.matmul(w) + (Tensor::ones({n_samples, out_dim}) * b);

    if (family_ == GLMFamily::Gaussian) {
        return eta;
    } else if (family_ == GLMFamily::Bernoulli) {
        return eta.sigmoid();
    } else if (family_ == GLMFamily::Multinomial) {
        return eta.softmax();
    } else {
        throw std::logic_error("Unsupported GLM family");
    }
}

float GLM::predict_one(const Tensor& x, const GLMResult& result) const {
    // accept x as 1-D [n_features] or 2-D [1, n_features]
    Tensor x_row = x;
    if (x.shape().size() == 1) {
        x_row = x.reshape({1, x.shape()[0]});
    } else if (x.shape().size() == 2) {
        if (x.shape()[0] != 1) throw std::invalid_argument("predict_one expects a single sample (shape [1, n_features])");
    } else {
        throw std::invalid_argument("predict_one: unsupported input shape");
    }

    Tensor pred = predict(x_row, result);
    // pred is [1, out_dim] - return a scalar:
    if (family_ == GLMFamily::Gaussian || family_ == GLMFamily::Bernoulli) {
        // return the single predicted value / probability
        return pred.data().at(0);
    } else if (family_ == GLMFamily::Multinomial) {
        // return predicted class index as float (argmax over last dim)
        const auto& pd = pred.data();
        const auto pshape = pred.shape();
        size_t out_dim = pshape.back();
        size_t offset = 0; // first (and only) row
        size_t argmax = 0;
        float best = pd[offset + 0];
        for (size_t i = 1; i < out_dim; ++i) {
            float v = pd[offset + i];
            if (v > best) {
                best = v;
                argmax = i;
            }
        }
        return static_cast<float>(argmax);
    } else {
        throw std::logic_error("Unsupported GLM family");
    }
}

float GLM::compute_loss(const Tensor& y_pred, const Tensor& y) const {
    const auto& pred = y_pred;
    const auto& target = y;
    const auto& p_data = pred.data();
    const auto& t_data = target.data();

    if (family_ == GLMFamily::Gaussian) {
        // 0.5 * MSE
        double sse = 0.0;
        if (p_data.size() != t_data.size()) throw std::invalid_argument("Shapes mismatch in compute_loss");
        for (size_t i = 0; i < p_data.size(); ++i) {
            double d = static_cast<double>(t_data[i]) - static_cast<double>(p_data[i]);
            sse += d * d;
        }
        return static_cast<float>(0.5 * sse / static_cast<double>(p_data.size()));
    } else if (family_ == GLMFamily::Bernoulli) {
        if (p_data.size() != t_data.size()) throw std::invalid_argument("Shapes mismatch in compute_loss");
        double acc = 0.0;
        for (size_t i = 0; i < p_data.size(); ++i) {
            double p = std::min(1.0 - 1e-12, std::max(1e-12, static_cast<double>(p_data[i])));
            double t = static_cast<double>(t_data[i]);
            acc += -(t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
        }
        return static_cast<float>(acc / static_cast<double>(p_data.size()));
    } else if (family_ == GLMFamily::Multinomial) {
        // y_pred: [n_samples, C], y: [n_samples, C] (one-hot or soft labels)
        const auto pshape = pred.shape();
        if (pshape.size() < 2) throw std::invalid_argument("Multinomial compute_loss needs 2-D predictions");
        size_t rows = pshape[0];
        size_t cols = pshape[1];
        if (t_data.size() != p_data.size()) throw std::invalid_argument("Shapes mismatch in compute_loss");
        double acc = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            size_t off = r * cols;
            for (size_t c = 0; c < cols; ++c) {
                double p = std::min(1.0 - 1e-12, std::max(1e-12, static_cast<double>(p_data[off + c])));
                double t = static_cast<double>(t_data[off + c]);
                if (t > 0.0) acc += -t * std::log(p);
            }
        }
        return static_cast<float>(acc / static_cast<double>(rows));
    } else {
        throw std::logic_error("Unsupported GLM family");
    }
}

float GLM::compute_r_squared(const Tensor& y_pred, const Tensor& y) const {
    if (family_ != GLMFamily::Gaussian) return std::numeric_limits<float>::quiet_NaN();
    const auto& p = y_pred.data();
    const auto& t = y.data();
    if (p.size() != t.size()) throw std::invalid_argument("Shapes mismatch in compute_r_squared");
    double mean_y = 0.0;
    for (float v : t) mean_y += v;
    mean_y /= static_cast<double>(t.size());
    double sst = 0.0;
    double sse = 0.0;
    for (size_t i = 0; i < t.size(); ++i) {
        double dy = static_cast<double>(t[i]) - mean_y;
        sst += dy * dy;
        double res = static_cast<double>(t[i]) - static_cast<double>(p[i]);
        sse += res * res;
    }
    if (sst <= 0.0) return 0.0f;
    return 1.0f - static_cast<float>(sse / sst);
}

float GLM::grad_norm(const Tensor& w_grad, float b_grad) const {
    double acc = 0.0;
    for (float v : w_grad.data()) acc += static_cast<double>(v) * static_cast<double>(v);
    acc += static_cast<double>(b_grad) * static_cast<double>(b_grad);
    return static_cast<float>(std::sqrt(acc));
}

} // namespace tiny_transformer
