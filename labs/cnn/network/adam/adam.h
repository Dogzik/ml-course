#pragma once

#include <vector>
#include "common_types.h"

segment update_moments_get_hats(segment &moments, const segment &grad, int t);
matrix update_moments_get_hats(matrix &moments, const matrix &grad, int t);
tensor update_moments_get_hats(tensor &moments, const tensor &grad, int t);

segment update_adaptive_get_hats(segment &adaptive, const segment &grad, int t);
matrix update_adaptive_get_hats(matrix &adaptive, const matrix &grad, int t);
tensor update_adaptive_get_hats(tensor &adaptive, const tensor &grad, int t);

void update_params(segment &moments, segment &adaptive, const segment &grad, segment &params, int t, double mu);
void update_params(matrix &moments, matrix &adaptive, const matrix &grad, matrix &params, int t, double mu);
void update_params(tensor &moments, tensor &adaptive, const tensor &grad, tensor &params, int t, double mu);
