#include <cmath>
#include <cassert>
#include "loss.h"
#include "softmax.h"


double cross_entropy_loss(const std::vector<double> &source, int real_label) {
  assert(0 <= real_label && static_cast<size_t>(real_label) < source.size());
  auto probabilities = softmax(source);
  return -std::log(probabilities[real_label]);
}

std::vector<double> cross_entropy_derivative(const std::vector<double> &source, int real_label) {
  assert(0 <= real_label && static_cast<size_t>(real_label) < source.size());
  auto res = softmax(source);
  res[real_label] -= 1;
  return res;
}
