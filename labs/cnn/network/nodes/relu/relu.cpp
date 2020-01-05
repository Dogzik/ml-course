#include "relu.h"

relu::relu(double inv_alpha, size<tensor> sizes) : inv_alpha{inv_alpha} {
  in_size = sizes;
  out_size = sizes;
}

tensor relu::compute(const tensor &input) {
  tensor value = input;
  input_copy = input;
  for (auto &layer : value) {
    for (auto &row : layer) {
      for (auto &cell : row) {
        if (cell < 0) {
          cell /= inv_alpha;
        }
      }
    }
  }
  return value;
}

tensor relu::do_backprop(const tensor &output_diff) {
  auto res = tensor(in_size.depth, matrix(in_size.height, segment(in_size.width, 0)));
  for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
    for (ptrdiff_t j = 0; j < out_size.height; ++j) {
      for (ptrdiff_t k = 0; k < out_size.width; ++k) {
        auto cur_input = input_copy.at(i).at(j).at(k);
        double multiplier;
        if (cur_input >= 0) {
          multiplier = 1.0;
        } else {
          multiplier = 1.0 / inv_alpha;
        }
        res.at(i).at(j).at(k) += multiplier * output_diff.at(i).at(j).at(k);
      }
    }
  }
  return res;
}
