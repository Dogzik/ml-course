#pragma once

#include <vector>

double cross_entropy_loss(const std::vector<double> &source, int real_label);
std::vector<double> cross_entropy_derivative(const std::vector<double> &source, int real_label);
