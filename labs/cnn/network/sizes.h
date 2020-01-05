#pragma once

#include "common_types.h"

template<typename T>
struct size;

template<>
struct size<tensor> {
  ptrdiff_t depth;
  ptrdiff_t height;
  ptrdiff_t width;

  [[nodiscard]] ptrdiff_t flatten_size() const noexcept {
    return depth * height * width;
  }
};

template<>
struct size<matrix> {
  ptrdiff_t height;
  ptrdiff_t width;

  [[nodiscard]] ptrdiff_t flatten_size() const noexcept {
    return height * width;
  }
};

template<>
struct size<segment> {
  ptrdiff_t width;

  [[nodiscard]] ptrdiff_t flatten_size() const noexcept {
    return width;
  }
};
