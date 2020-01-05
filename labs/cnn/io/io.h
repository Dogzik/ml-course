#pragma once

#include <vector>
#include <string>
#include "common_types.h"

std::vector<tensor> read_mnist_images(const std::string &filename);
std::vector<int> read_mnist_labels(const std::string &filename);
