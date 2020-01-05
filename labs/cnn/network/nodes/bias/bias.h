#pragma once

#include "generators.h"
#include "nodes/node.h"

struct bias : node<tensor, tensor> {
  segment b;
  segment db;

  segment moments;
  segment adaptive;

  explicit bias(size<tensor> sizes);

  tensor compute(const tensor &input) final;
  tensor do_backprop(const tensor &output_diff) final;
  void update_grad(int t, double mu) final;
};
