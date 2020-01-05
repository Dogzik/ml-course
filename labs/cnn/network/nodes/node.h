#pragma once

#include <vector>
#include <cstddef>
#include "common_types.h"
#include "sizes.h"

template<typename IN, typename OUT>
struct node {
  size<IN> in_size;
  size<OUT> out_size;
  int step;

  node() : step{1} {}

  virtual OUT compute(const IN &input) = 0;
  virtual IN do_backprop(const OUT &output_diff) = 0;
  virtual void update_grad(int t, double mu = 1e-3) = 0;

  IN backprop(const OUT &output_diff) {
    auto res = do_backprop(output_diff);
    update_grad(step++);
    return res;
  }

  virtual ~node() = default;
};
