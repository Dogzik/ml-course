#pragma once

#include "nodes/cnv/cnv.h"

struct cnve : cnv {
  using cnv::cnv;

  matrix get_padded_layer(const matrix &layer) final;
  void compress_padded_diff_layer(matrix &layer) final;
};
