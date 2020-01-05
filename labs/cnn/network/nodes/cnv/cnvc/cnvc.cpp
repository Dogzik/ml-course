#include "cnvc.h"

matrix cnvc::get_padded_layer(const matrix &layer) {
  ptrdiff_t n = layer.size();
  ptrdiff_t m = layer.at(0).size();
  matrix result(n + 2 * p, segment(m + 2 * p));
  for (ptrdiff_t i = 0; i < n; ++i) {
    for (ptrdiff_t j = 0; j < m; ++j) {
      result.at(i + p).at(j + p) = layer.at(i).at(j);
    }
  }

  for (ptrdiff_t i = p; i < p + n; ++i) {
    for (ptrdiff_t j = p - 1; j >= 0; --j) {
      result.at(i).at(j) = result.at(i).at(j + m);
    }
    for (ptrdiff_t j = m + p; j < m + 2 * p; ++j) {
      result.at(i).at(j) = result.at(i).at(j - m);
    }
  }
  for (ptrdiff_t i = p - 1; i >= 0; --i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      result.at(i).at(j) = result.at(i + n).at(j);
    }
  }
  for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      result.at(i).at(j) = result.at(i - n).at(j);
    }
  }
  return result;
}

void cnvc::compress_padded_diff_layer(matrix &layer) {
  ptrdiff_t n = layer.size() - 2 * p;
  ptrdiff_t m = layer.at(0).size() - 2 * p;
  for (long long i = p - 1; i >= 0; --i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      layer.at(i + n).at(j) += layer.at(i).at(j);
    }
  }
  for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      layer.at(i - n).at(j) += layer.at(i).at(j);
    }
  }

  for (ptrdiff_t i = p; i < p + n; ++i) {
    for (long long j = p - 1; j >= 0; --j) {
      layer.at(i).at(j + m) += layer.at(i).at(j);
    }
    for (ptrdiff_t j = m + p; j < m + 2 * p; ++j) {
      layer.at(i).at(j - m) += layer.at(i).at(j);
    }
  }
}
