#include <iostream>
#include <algorithm>
#include "softmax/loss.h"
#include "softmax/softmax.h"
#include "network.h"
#include "nodes/bias/bias.h"
#include "nodes/relu/relu.h"
#include "nodes/pool/pool.h"
#include "nodes/cnv/cnvm/cnvm.h"
#include "nodes/cnv/cnve/cnve.h"
#include "nodes/cnv/cnvc/cnvc.h"

network::network(size<tensor> sizes) : input_size{sizes} {}

size<tensor> network::get_cur_sizes() const {
  if (nodes.empty()) {
    return input_size;
  } else {
    return nodes.back()->out_size;
  }
}

void network::add_bias() {
  nodes.push_back(std::make_unique<bias>(get_cur_sizes()));
}

void network::add_relu(double inv_alpha) {
  nodes.push_back(std::make_unique<relu>(inv_alpha, get_cur_sizes()));
}

void network::add_pool(ptrdiff_t s) {
  nodes.push_back(std::make_unique<pool>(s, get_cur_sizes()));
}

void network::add_cnv(size_t cnt, size_t n, size_t m, ptrdiff_t p, ptrdiff_t s, CNV_TYPE padding) {
  switch (padding) {
    case CNV_TYPE::MIRROR: {
      nodes.push_back(std::make_unique<cnvm>(cnt, n, m, p, s, get_cur_sizes()));
      break;
    }
    case CNV_TYPE::EXTEND: {
      nodes.push_back(std::make_unique<cnve>(cnt, n, m, p, s, get_cur_sizes()));
      break;
    }
    case CNV_TYPE::CYCLIC: {
      nodes.push_back(std::make_unique<cnvc>(cnt, n, m, p, s, get_cur_sizes()));
      break;
    }
  }
}

void network::add_flat() {
  transformer = std::make_unique<flat>(get_cur_sizes());
}

void network::add_fulcon(ptrdiff_t labels_cnt) {
  final_node = std::make_unique<fulcon>(labels_cnt, transformer->out_size);
}

void network::finish_net(ptrdiff_t labels_cnt) {
  add_flat();
  add_fulcon(labels_cnt);
}

double network::process_image(const tensor &image, int real_label) {
  tensor value = image;
  for (auto &item : nodes) {
    value = item->compute(value);
  }
  segment flat_value = transformer->compute(value);
  flat_value = final_node->compute(flat_value);

  auto diff = cross_entropy_derivative(flat_value, real_label);

  diff = final_node->backprop(diff);
  auto tensor_diff = transformer->backprop(diff);
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    tensor_diff = it->get()->backprop(tensor_diff);
  }
  return cross_entropy_loss(flat_value, real_label);
}

void network::fit_epoch(const std::vector<tensor> &images, const std::vector<int> &real_labels) {
  double loss = 0;
  int k = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    loss += process_image(images[i], real_labels[i]);
    ++k;
    if (k == 1000) {
      std::cout << loss << std::endl;
      k = 0;
      loss = 0;
    }
  }
}

int network::predict_image(const tensor &image) {
  tensor value = image;
  for (auto &item : nodes) {
    value = item->compute(value);
  }
  segment flat_value = transformer->compute(value);
  flat_value = final_node->compute(flat_value);
  auto probabilities = softmax(flat_value);
  return std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
}
