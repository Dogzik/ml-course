#pragma once

#include "nodes/node.h"

struct flat : node<tensor, segment> {
  explicit flat(size<tensor> input_sizes);

  segment compute(const tensor &input) final;
  tensor do_backprop(const segment &output_diff) final;
  void update_grad(int, double) final {};
};
