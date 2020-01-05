#pragma once

#include "generators.h"
#include "nodes/node.h"

struct fulcon : node<segment , segment>{
  matrix weights;
  matrix dweights;

  matrix moments;
  matrix adaptives;

  segment input_copy;

  fulcon(ptrdiff_t labels_cnt, size<segment> input_sizes);

  segment compute(const segment &input) final;
  segment do_backprop(const segment &output_diff) final;
  void update_grad(int t, double mu) final;
};
