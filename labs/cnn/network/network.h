#pragma once

#include <vector>
#include <memory>

#include "nodes/node.h"
#include "nodes/fulcon/fulcon.h"
#include "nodes/flat/flat.h"
#include "nodes/cnv/cnv.h"

struct network {
  size<tensor> input_size;
  std::vector<std::unique_ptr<node<tensor, tensor>>> nodes;
  std::unique_ptr<flat> transformer;
  std::unique_ptr<fulcon> final_node;

  explicit network(size<tensor> sizes);

  size<tensor> get_cur_sizes() const;

  void add_bias();
  void add_relu(double inv_alpha);
  void add_pool(ptrdiff_t s);
  void add_cnv(size_t cnt, size_t n, size_t m, ptrdiff_t p, ptrdiff_t s, CNV_TYPE padding);
  void add_flat();
  void add_fulcon(ptrdiff_t labels_cnt);
  void finish_net(ptrdiff_t labels_cnt);


  double process_image(const tensor &image, int real_label);
  void fit_epoch(const std::vector<tensor> &images, const  std::vector<int> &real_labels);
  int predict_image(const tensor &image);
};
