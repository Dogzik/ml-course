#pragma once

#include "generators.h"
#include "nodes/node.h"

enum class CNV_TYPE {
  MIRROR,
  EXTEND,
  CYCLIC
};

struct cnv : node<tensor, tensor> {
  std::vector<tensor> kernels;
  std::vector<tensor> dkernels;
  ptrdiff_t p;
  ptrdiff_t s;
  tensor padded_input;

  std::vector<tensor> moments;
  std::vector<tensor> adaptive;

  cnv(size_t cnt, size_t n, size_t m, ptrdiff_t p, ptrdiff_t s, size<tensor> input_sizes);


  virtual matrix get_padded_layer(const matrix &layer) = 0;
  virtual void compress_padded_diff_layer(matrix &layer) = 0;

  tensor compute(const tensor &input) final;
  tensor do_backprop(const tensor &output_diff) final;
  void update_grad(int t, double mu) final;

  tensor get_padded_backprop(const tensor &output_diff);
};
