#include <limits>
#include "pool.h"

pool::pool(ptrdiff_t s, size <tensor> input_sizes) : s{s} {
  in_size = input_sizes;
  out_size.depth = in_size.depth;
  out_size.height = (in_size.height - s) / s + 1;
  out_size.width = (in_size.width - s) / s + 1;
}

tensor pool::compute(const tensor &input) {
  auto value = tensor(out_size.depth,
                      matrix(out_size.height,
                             segment(out_size.width, -std::numeric_limits<double>::infinity())));
  input_copy = input;
  for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
    for (ptrdiff_t j = 0; j < out_size.height; ++j) {
      for (ptrdiff_t k = 0; k < out_size.width; ++k) {
        for (ptrdiff_t dj = 0; dj < s; ++dj) {
          for (ptrdiff_t dk = 0; dk < s; ++dk) {
            value.at(i).at(j).at(k) = std::max(value.at(i).at(j).at(k), input.at(i).at(j * s + dj).at(k * s + dk));
          }
        }
      }
    }
  }
  return value;
}

tensor pool::do_backprop(const tensor &output_diff) {
  auto res = tensor(in_size.depth, matrix(in_size.height, segment(in_size.width, 0)));
  for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
    for (ptrdiff_t j = 0; j < out_size.height; ++j) {
      for (ptrdiff_t k = 0; k < out_size.width; ++k) {
        double cur_max = -std::numeric_limits<double>::infinity();
        for (ptrdiff_t dj = 0; dj < s; ++dj) {
          for (ptrdiff_t dk = 0; dk < s; ++dk) {
            cur_max = std::max(cur_max, input_copy.at(i).at(j * s + dj).at(k * s + dk));
          }
        }
        for (ptrdiff_t dj = 0; dj < s; ++dj) {
          for (ptrdiff_t dk = 0; dk < s; ++dk) {
            if (input_copy.at(i).at(j * s + dj).at(k * s + dk) == cur_max) {
              res.at(i).at(j * s + dj).at(k * s + dk) += output_diff.at(i).at(j).at(k);
            }
          }
        }
      }
    }
  }
  return res;
}
