#include "adam/adam.h"
#include "cnv.h"

cnv::cnv(size_t cnt, size_t n, size_t m, ptrdiff_t p, ptrdiff_t s, size<tensor> input_sizes) :
        kernels(cnt),
        p{p}, s{s} {
  in_size = input_sizes;
  out_size.depth = cnt;
  out_size.height = (in_size.height + 2 * p - n) / s + 1;
  out_size.width = (in_size.width + 2 * p - m) / s + 1;
  for (auto &kernel: kernels) {
    //kernel = tensor(in_size.depth, matrix(n, segment(m, 1)));
    kernel = gen_tensor(in_size.depth, n, m);
  }
  moments.assign(cnt, tensor(in_size.depth, matrix(n, segment(m, 0))));
  adaptive = moments;
}

tensor cnv::compute(const tensor &input) {
  ptrdiff_t d = in_size.depth;
  ptrdiff_t k1 = kernels.at(0).at(0).size();
  ptrdiff_t k2 = kernels.at(0).at(0).at(0).size();

  padded_input.resize(d);
  for (ptrdiff_t i = 0; i < d; ++i) {
    padded_input.at(i) = get_padded_layer(input.at(i));
  }
  auto value = tensor(out_size.depth, matrix(out_size.height, segment(out_size.width, 0)));

  for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
    for (ptrdiff_t cur_d = 0; cur_d < d; ++cur_d) {
      for (ptrdiff_t j = 0; j < out_size.height; ++j) {
        for (ptrdiff_t k = 0; k < out_size.width; ++k) {
          // here we have one cell of output
          for (ptrdiff_t dj = 0; dj < k1; ++dj) {
            for (ptrdiff_t dk = 0; dk < k2; ++dk) {
              value.at(i).at(j).at(k) +=
                      padded_input.at(cur_d).at(j * s + dj).at(k * s + dk) * kernels.at(i).at(cur_d).at(dj).at(dk);
            }
          }
        }
      }
    }
  }
  dkernels.assign(kernels.size(), tensor(d, matrix(k1, segment(k2, 0))));
  return value;
}

tensor cnv::get_padded_backprop(const tensor &output_diff) {
  ptrdiff_t d = in_size.depth;
  ptrdiff_t n = in_size.height;
  ptrdiff_t m = in_size.width;
  ptrdiff_t k1 = kernels.at(0).at(0).size();
  ptrdiff_t k2 = kernels.at(0).at(0).at(0).size();
  tensor padded_input_diff(d, matrix(n + 2 * p, segment(m + 2 * p, 0)));

  for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
    for (ptrdiff_t cur_d = 0; cur_d < d; ++cur_d) {
      for (ptrdiff_t j = 0; j < out_size.height; ++j) {
        for (ptrdiff_t k = 0; k < out_size.width; ++k) {
          for (ptrdiff_t dj = 0; dj < k1; ++dj) {
            for (ptrdiff_t dk = 0; dk < k2; ++dk) {
              padded_input_diff.at(cur_d).at(j * s + dj).at(k * s + dk) +=
                      kernels.at(i).at(cur_d).at(dj).at(dk) * output_diff.at(i).at(j).at(k);
              dkernels.at(i).at(cur_d).at(dj).at(dk) +=
                      padded_input.at(cur_d).at(j * s + dj).at(k * s + dk) * output_diff.at(i).at(j).at(k);
            }
          }
        }
      }
    }
  }
  return padded_input_diff;
}

tensor cnv::do_backprop(const tensor &output_diff) {
  auto padded_input_diff = get_padded_backprop(output_diff);
  ptrdiff_t d = in_size.depth;
  ptrdiff_t n = in_size.height;
  ptrdiff_t m = in_size.width;
  tensor res(d, matrix(n, segment(m, 0)));
  for (ptrdiff_t i = 0; i < d; ++i) {
    compress_padded_diff_layer(padded_input_diff.at(i));
    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t k = 0; k < m; ++k) {
        res.at(i).at(j).at(k) += padded_input_diff.at(i).at(j + p).at(k + p);
      }
    }
  }
  return res;
}

void cnv::update_grad(int t, double mu) {
  for (size_t i = 0; i < dkernels.size(); ++i) {
    update_params(moments[i], adaptive[i], dkernels[i], kernels[i], t, mu);
  }
}
