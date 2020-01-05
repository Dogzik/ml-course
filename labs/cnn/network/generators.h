#pragma once

#include <cstddef>
#include "common_types.h"

double gen_double();
segment gen_segment(size_t size);
matrix gen_matrix(size_t n, size_t m);
tensor gen_tensor(size_t d, size_t n, size_t m);
