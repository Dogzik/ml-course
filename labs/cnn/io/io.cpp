#include "io.h"

#include <fstream>

namespace {
  int shift_left(unsigned char t, int cnt) {
    return static_cast<int>(t) << cnt;
  }

  int read_int(std::ifstream &file) {
    char buff[4]{};
    file.read(buff, 4);
    return shift_left(buff[0], 24) + shift_left(buff[1], 16) + shift_left(buff[2], 8) + buff[3];
  }
}


std::vector<tensor> read_mnist_images(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);

  std::vector<std::vector<double>> arr;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;
  file.ignore(4); // for magic number
  number_of_images = read_int(file);
  n_rows = read_int(file);
  n_cols = read_int(file);
  arr.resize(number_of_images, std::vector<double>(n_rows * n_cols));
  std::vector<tensor> res(number_of_images, tensor(1, matrix(n_rows, segment(n_cols))));
  for (int i = 0; i < number_of_images; ++i) {
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_cols; ++c) {
        unsigned char temp = 0;
        file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
        res[i][0][r][c] = temp / 255.0;
      }
    }
  }
  return res;
}

std::vector<int> read_mnist_labels(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  int number_of_labels = 0;
  file.ignore(4); // for magic
  number_of_labels = read_int(file);
  std::vector<int> arr(number_of_labels);
  for (int i = 0; i < number_of_labels; ++i) {
    unsigned char temp = 0;
    file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
    arr[i] = temp;
  }
  return arr;
}
