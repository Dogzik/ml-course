#include "flat.h"

flat::flat(size<tensor> input_sizes) {
  in_size = input_sizes;
  out_size.width = in_size.flatten_size();
}

segment flat::compute(const tensor &input) {
  auto d = in_size.depth;
  auto n = in_size.height;
  auto m = in_size.width;
  segment value(out_size.width);
  for (ptrdiff_t i = 0; i < d; ++i) {
    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t k = 0; k < m; ++k) {
        value.at(i * (n * m) + j * m + k) = input.at(i).at(j).at(k);
      }
    }
  }
  return value;
}

tensor flat::do_backprop(const segment &output_diff) {
  auto d = in_size.depth;
  auto n = in_size.height;
  auto m = in_size.width;
  tensor res(d, matrix(n, segment(m)));
  for (ptrdiff_t i = 0; i < d; ++i) {
    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t k = 0; k < m; ++k) {
         res.at(i).at(j).at(k) = output_diff.at(i * (n * m) + j * m + k);
      }
    }
  }
  return res;
}
