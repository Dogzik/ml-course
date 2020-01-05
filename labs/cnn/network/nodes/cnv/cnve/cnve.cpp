#include "cnve.h"

matrix cnve::get_padded_layer(const matrix &layer) {
  ptrdiff_t n = layer.size();
  ptrdiff_t m = layer.at(0).size();
  matrix result(n + 2 * p, segment(m + 2 * p));
  for (ptrdiff_t i = 0; i < n; ++i) {
    for (ptrdiff_t j = 0; j < m; ++j) {
      result.at(i + p).at(j + p) = layer.at(i).at(j);
    }
  }
  for (ptrdiff_t i = p; i < p + n; ++i) {
    for (ptrdiff_t j = 0; j < p; ++j) {
      result.at(i).at(j) = layer.at(i - p).front();
    }
    for (ptrdiff_t j = m + p; j < m + 2 * p; ++j) {
      result.at(i).at(j) = layer.at(i - p).back();
    }
  }
  for (ptrdiff_t i = 0; i < p; ++i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      result.at(i).at(j) = result.at(p).at(j);
    }
  }
  for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      result.at(i).at(j) = result.at(n + p - 1).at(j);
    }
  }
  return result;
}

void cnve::compress_padded_diff_layer(matrix &layer) {
  ptrdiff_t n = layer.size() - 2 * p;
  ptrdiff_t m = layer.at(0).size() - 2 * p;

  for (ptrdiff_t i = 0; i < p; ++i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      layer.at(p).at(j) += layer.at(i).at(j);
    }
  }
  for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
    for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
      layer.at(n + p - 1).at(j) += layer.at(i).at(j);
    }
  }

  for (ptrdiff_t i = p; i < p + n; ++i) {
    for (ptrdiff_t j = 0; j < p; ++j) {
      layer.at(i).at(p) += layer.at(i).at(j);
    }
    for (ptrdiff_t j = m + p; j < m + 2 * p; ++j) {
      layer.at(i).at(m + p - 1) += layer.at(i).at(j);
    }
  }
}
