#include <cmath>
#include "adam.h"

namespace {
  constexpr double B1 = 0.9;
  constexpr double B2 = 0.999;
  constexpr double EPS = 1e-6;
}

segment update_moments_get_hats(segment &moments, const segment &grad, int t) {
  double DIVISOR = 1.0 - std::pow(B1, t);
  segment hats(moments.size());
  for (size_t i = 0; i < moments.size(); ++i) {
    moments[i] = B1 * moments[i] + (1 - B1) * grad[i];
    hats[i] = moments[i] / DIVISOR;
  }
  return hats;
}

matrix update_moments_get_hats(matrix &moments, const matrix &grad, int t) {
  matrix hats(moments.size());
  for (size_t i = 0; i < moments.size(); ++i) {
    hats[i] = update_moments_get_hats(moments[i], grad[i], t);
  }
  return hats;
}

tensor update_moments_get_hats(tensor &moments, const tensor &grad, int t) {
  tensor hats(moments.size());
  for (size_t i = 0; i < moments.size(); ++i) {
    hats[i] = update_moments_get_hats(moments[i], grad[i], t);
  }
  return hats;
}

segment update_adaptive_get_hats(segment &adaptive, const segment &grad, int t) {
  double DIVISOR = 1 - std::pow(B2, t);
  segment hats(adaptive.size());
  for (size_t i = 0; i < adaptive.size(); ++i) {
    adaptive[i] = B2 * adaptive[i] + (1 - B2) * grad[i] * grad[i];
    hats[i] = adaptive[i] / DIVISOR;
  }
  return hats;
}

matrix update_adaptive_get_hats(matrix &adaptive, const matrix &grad, int t) {
  matrix hats(adaptive.size());
  for (size_t i = 0; i < adaptive.size(); ++i) {
    hats[i] = update_adaptive_get_hats(adaptive[i], grad[i], t);
  }
  return hats;
}

tensor update_adaptive_get_hats(tensor &adaptive, const tensor &grad, int t) {
  tensor hats(adaptive.size());
  for (size_t i = 0; i < adaptive.size(); ++i) {
    hats[i] = update_adaptive_get_hats(adaptive[i], grad[i], t);
  }
  return hats;
}

void update_params(segment &moments, segment &adaptive, const segment &grad, segment &params, int t, double mu) {
  auto hat_moments = update_moments_get_hats(moments, grad, t);
  auto hat_adaptive = update_adaptive_get_hats(adaptive, grad, t);
  for (size_t i = 0; i < grad.size(); ++i) {
    double diff = mu * hat_moments[i] / (std::sqrt(hat_adaptive[i]) + EPS);
    params[i] -= diff;
  }
}

void update_params(matrix &moments, matrix &adaptive, const matrix &grad, matrix &params, int t, double mu) {
  for (size_t i = 0; i < grad.size(); ++i) {
    update_params(moments[i], adaptive[i], grad[i], params[i], t, mu);
  }
}

void update_params(tensor &moments, tensor &adaptive, const tensor &grad, tensor &params, int t, double mu) {
  for (size_t i = 0; i < grad.size(); ++i) {
    update_params(moments[i], adaptive[i], grad[i], params[i], t, mu);
  }
}


