#include "softmax.h"

#include <cmath>
#include <numeric>

std::vector<double> softmax(const std::vector<double> &source) {
  auto fix = std::accumulate(source.begin(), source.end(), 0.0) / source.size();
  auto res = source;
  double sum = 0.0;
  for (auto &x : res) {
    double exp_x = std::exp(x - fix);
    sum += exp_x;
    x = exp_x;
  }
  for (auto &x : res) {
    x /= sum;
  }
  return res;
}
