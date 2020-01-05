#pragma once

#include "nodes/node.h"

struct pool : node<tensor, tensor> {
  ptrdiff_t s;
  tensor input_copy;

  pool(ptrdiff_t s, size<tensor> input_sizes);

  tensor compute(const tensor& input) final;
  tensor do_backprop(const tensor& output_diff) final;
  void update_grad(int, double) final {};
};
