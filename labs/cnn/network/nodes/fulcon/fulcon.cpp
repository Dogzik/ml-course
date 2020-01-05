#include "adam/adam.h"
#include "fulcon.h"

fulcon::fulcon(ptrdiff_t labels_cnt, size<segment> input_sizes) {
  in_size = input_sizes;
  out_size.width = labels_cnt;
  weights = gen_matrix(labels_cnt, in_size.width);
  moments.assign(labels_cnt, segment(in_size.width, 0));
  adaptives = moments;
  dweights = moments;
}

segment fulcon::compute(const segment &input) {
  input_copy = input;
  segment value(out_size.width, 0);
  for (ptrdiff_t i = 0; i < out_size.width; ++i) {
    for (ptrdiff_t j = 0; j < in_size.width; ++j) {
      value.at(i) += input.at(j) * weights.at(i).at(j);
    }
  }
  dweights.assign(out_size.width, segment(in_size.width, 0));
  return value;
}

segment fulcon::do_backprop(const segment &output_diff) {
  segment res(in_size.width, 0);
  for (ptrdiff_t i = 0; i < out_size.width; ++i) {
    for (ptrdiff_t j = 0; j < in_size.width; ++j) {
      res.at(j) += output_diff.at(i) * weights.at(i).at(j);
      dweights.at(i).at(j) += output_diff.at(i) * input_copy.at(j);
    }
  }
  return res;
}

void fulcon::update_grad(int t, double mu) {
  update_params(moments, adaptives, dweights, weights, t, mu);
}
