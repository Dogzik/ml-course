#include "bias.h"
#include "adam/adam.h"

bias::bias(size<tensor> sizes) : b{gen_segment(sizes.depth)} {
  in_size = sizes;
  out_size = sizes;
  moments.assign(b.size(), 0);
  adaptive.assign(b.size(), 0);
}

tensor bias::compute(const tensor &input) {
  auto value = input;
  for (ptrdiff_t i = 0; i < in_size.depth; ++i) {
    for (auto &row : value.at(i)) {
      for (auto &cell : row) {
        cell += b.at(i);
      }
    }
  }
  return value;
}

tensor bias::do_backprop(const tensor &output_diff) {
  auto res = tensor(in_size.depth, matrix(in_size.height, segment(in_size.width, 0)));
  db.assign(b.size(), 0);
  for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
    for (ptrdiff_t j = 0; j < out_size.height; ++j) {
      for (ptrdiff_t k = 0; k < out_size.width; ++k) {
        res.at(i).at(j).at(k) += output_diff.at(i).at(j).at(k);
        db.at(i) += output_diff.at(i).at(j).at(k);
      }
    }
  }
  return res;
}

void bias::update_grad(int t, double mu) {
  update_params(moments, adaptive, db, b, t, mu);
}
