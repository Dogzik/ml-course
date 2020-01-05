#include <random>
#include <chrono>
#include "generators.h"

double gen_double() {
  static auto generator = std::default_random_engine(time(nullptr));
  static auto distribution = std::uniform_real_distribution<double>(-1, 1);
  return distribution(generator);
}

segment gen_segment(size_t size) {
  segment res(size);
  for (auto &x : res) {
    x = gen_double();
  }
  return res;
}

matrix gen_matrix(size_t n, size_t m) {
  matrix res(n);
  for (auto &x : res) {
    x = gen_segment(m);
  }
  return res;
}

tensor gen_tensor(size_t d, size_t n, size_t m) {
  tensor res(d);
  for (auto &x : res) {
    x = gen_matrix(n, m);
  }
  return res;
}
