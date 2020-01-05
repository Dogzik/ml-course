#include <string>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include "network.h"
#include "io/io.h"

/**
 * See
 * http://yann.lecun.com/exdb/mnist/
 * https://github.com/zalandoresearch/fashion-mnist#get-the-data
 */
const std::string DATASET_PATH = "../dataset/";
const std::string TRAIN_DATA = "train-images.idx3-ubyte";
const std::string TEST_DATA = "t10k-images.idx3-ubyte";
const std::string TRAIN_LABELS = "train-labels.idx1-ubyte";
const std::string TEST_LABELS = "t10k-labels.idx1-ubyte";

network get_network(size_t labels_cnt, ptrdiff_t height, ptrdiff_t width) {
  network net({1, height, width});

  net.add_cnv(8, 3, 3, 1, 1, CNV_TYPE::EXTEND);
  net.add_relu(10);
  net.add_bias();
  net.add_pool(2);
  net.add_cnv(4, 3, 3, 1, 1, CNV_TYPE::MIRROR);
  net.add_relu(10);
  net.add_bias();
  net.add_pool(2);
  net.finish_net(labels_cnt);
  return net;
}

double calc_accuracy(std::vector<int> y_true, std::vector<int> y_pred) {
  ptrdiff_t good_cnt = 0;
  for (size_t i = 0; i < y_true.size(); ++i) {
    good_cnt += (y_true[i] == y_pred[i]);
  }
  return static_cast<double>(good_cnt) / y_true.size();
}

void shuffle_input(std::vector<tensor> &images, std::vector<int> &labels) {
  assert(images.size() == labels.size());
  auto sz = labels.size();
  std::vector<size_t> idxs(sz);
  std::iota(idxs.begin(), idxs.end(), 0);
  std::shuffle(idxs.begin(), idxs.end(), std::default_random_engine(228));

  std::vector<tensor> new_images;
  std::vector<int> new_labels;
  new_images.reserve(sz);
  new_labels.reserve(sz);
  for (auto idx : idxs) {
    new_images.push_back(std::move(images[idx]));
    new_labels.push_back(labels[idx]);
  }
  images = std::move(new_images);
  labels = std::move(new_labels);
}

int main() {
  auto train_data = read_mnist_images(DATASET_PATH + DIGITS + TRAIN_DATA);
  auto train_labels = read_mnist_labels(DATASET_PATH + DIGITS + TRAIN_LABELS);
  std::cout << "Train data loaded\n";

  auto height = train_data[0][0].size();
  auto width = train_data[0][0][0].size();
  auto labels_cnt = *std::max_element(train_labels.begin(), train_labels.end()) + 1;
  network net = get_network(labels_cnt, height, width);


  for (size_t i = 0; i < 3; ++i) {
    shuffle_input(train_data, train_labels);
    net.fit_epoch(train_data, train_labels);
    std::cout << "Fitted " + std::to_string(i + 1) << " epochs\n";
  }

  auto test_data = read_mnist_images(DATASET_PATH + DIGITS + TEST_DATA);
  auto test_labels = read_mnist_labels(DATASET_PATH + DIGITS + TEST_LABELS);
  std::cout << "Test data loaded\n";

  std::vector<int> pred;
  pred.reserve(test_labels.size());
  for (const auto &image : test_data) {
    pred.push_back(net.predict_image(image));
  }
  std::cout << "Finished predicting\n";
  std::cout << std::fixed << "Accuracy: " << calc_accuracy(test_labels, pred) << std::endl;
  return 0;
}
