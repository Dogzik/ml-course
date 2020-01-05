#pragma once

#include "nodes/node.h"

struct relu : node<tensor, tensor> {
  double inv_alpha;
  tensor input_copy;

  relu(double inv_alpha, size<tensor> sizes);

  tensor compute(const tensor& input) final;
  tensor do_backprop(const tensor& output_diff) final;
  void update_grad(int, double) final {};
};
