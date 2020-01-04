#include <vector>
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <algorithm>

using segment = std::vector<double>;
using matrix = std::vector<segment>;
using tensor = std::vector<matrix>;

void print_tensor(const tensor &data) {
  for (auto &layer : data) {
    for (auto &row : layer) {
      for (auto item : row) {
        std::cout << item << " ";
      }
    }
  }
  std::cout << std::endl;
}

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

template<typename IN, typename OUT>
struct node {
  size<IN> in_size;
  size<OUT> out_size;
  int step;

  node() : step{1} {}

  virtual OUT compute(const IN &input) = 0;

  virtual IN do_backprop(const OUT &output_diff) = 0;

  IN backprop(const OUT &output_diff) {
    auto res = do_backprop(output_diff);
    //update_grad(step++);
    return res;
  }

  virtual void print_parameters_diff() = 0;
  virtual ~node() = default;
};

struct relu : node<tensor, tensor> {
  double inv_alpha;
  tensor input_copy;

  relu(double inv_alpha, size<tensor> sizes) : inv_alpha{inv_alpha} {
    in_size = sizes;
    out_size = sizes;
  }

  tensor compute(const tensor &input) final {
    tensor value = input;
    input_copy = input;
    for (auto &layer : value) {
      for (auto &row : layer) {
        for (auto &cell : row) {
          if (cell < 0) {
            cell /= inv_alpha;
          }
        }
      }
    }
    return value;
  }

  tensor do_backprop(const tensor &output_diff) final {
    auto res = tensor(in_size.depth, matrix(in_size.height, segment(in_size.width, 0)));
    for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
      for (ptrdiff_t j = 0; j < out_size.height; ++j) {
        for (ptrdiff_t k = 0; k < out_size.width; ++k) {
          auto cur_input = input_copy.at(i).at(j).at(k);
          double multiplier;
          if (cur_input >= 0) {
            multiplier = 1.0;
          } else {
            multiplier = 1.0 / inv_alpha;
          }
          res.at(i).at(j).at(k) += multiplier * output_diff.at(i).at(j).at(k);
        }
      }
    }
    return res;
  }

  void print_parameters_diff() final {}
};

struct pool : node<tensor, tensor> {
  ptrdiff_t s;
  tensor input_copy;

  pool(ptrdiff_t s, size<tensor> input_sizes) : s{s} {
    in_size = input_sizes;
    out_size.depth = in_size.depth;
    out_size.height = (in_size.height - s) / s + 1;
    out_size.width = (in_size.width - s) / s + 1;
  }

  tensor compute(const tensor &input) final {
    auto value = tensor(out_size.depth,
                        matrix(out_size.height,
                               segment(out_size.width, -std::numeric_limits<double>::infinity())));
    input_copy = input;
    for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
      for (ptrdiff_t j = 0; j < out_size.height; ++j) {
        for (ptrdiff_t k = 0; k < out_size.width; ++k) {
          for (ptrdiff_t dj = 0; dj < s; ++dj) {
            for (ptrdiff_t dk = 0; dk < s; ++dk) {
              value.at(i).at(j).at(k) = std::max(value.at(i).at(j).at(k), input.at(i).at(j * s + dj).at(k * s + dk));
            }
          }
        }
      }
    }
    return value;
  }

  tensor do_backprop(const tensor &output_diff) final {
    auto res = tensor(in_size.depth, matrix(in_size.height, segment(in_size.width, 0)));
    for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
      for (ptrdiff_t j = 0; j < out_size.height; ++j) {
        for (ptrdiff_t k = 0; k < out_size.width; ++k) {
          double cur_max = -std::numeric_limits<double>::infinity();
          for (ptrdiff_t dj = 0; dj < s; ++dj) {
            for (ptrdiff_t dk = 0; dk < s; ++dk) {
              cur_max = std::max(cur_max, input_copy.at(i).at(j * s + dj).at(k * s + dk));
            }
          }
          for (ptrdiff_t dj = 0; dj < s; ++dj) {
            for (ptrdiff_t dk = 0; dk < s; ++dk) {
              if (input_copy.at(i).at(j * s + dj).at(k * s + dk) == cur_max) {
                res.at(i).at(j * s + dj).at(k * s + dk) += output_diff.at(i).at(j).at(k);
              }
            }
          }
        }
      }
    }
    return res;
  }

  void print_parameters_diff() final {}
};

struct bias : node<tensor, tensor> {
  segment b;
  segment db;

  segment moments;
  segment adaptive;

  explicit bias(size<tensor> sizes) : b(sizes.depth) {
    for (auto &item : b) {
      int x;
      std::cin >> x;
      item = x;
    }
    in_size = sizes;
    out_size = sizes;
    moments.assign(b.size(), 0);
    adaptive.assign(b.size(), 0);
  }

  tensor compute(const tensor &input) final {
    auto value = input;
    for (ptrdiff_t i = 0; i < in_size.depth; ++i) {
      for (auto &row : value.at(i)) {
        for (auto &cell : row) {
          cell += b.at(i);
        }
      }
    }
    return value;
  }

  tensor do_backprop(const tensor &output_diff) final {
    auto res = tensor(in_size.depth, matrix(in_size.height, segment(in_size.width, 0)));
    db.assign(b.size(), 0);
    for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
      for (ptrdiff_t j = 0; j < out_size.height; ++j) {
        for (ptrdiff_t k = 0; k < out_size.width; ++k) {
          res.at(i).at(j).at(k) += output_diff.at(i).at(j).at(k);
          db.at(i) += output_diff.at(i).at(j).at(k);
        }
      }
    }
    return res;
  }

  void print_parameters_diff() final {
    for (auto x : db) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  }
};

enum class CNV_TYPE {
  MIRROR,
  EXTEND,
  CYCLIC
};

struct cnv : node<tensor, tensor> {
  std::vector<tensor> kernels;
  std::vector<tensor> dkernels;
  ptrdiff_t p;
  ptrdiff_t s;
  tensor padded_input;

  std::vector<tensor> moments;
  std::vector<tensor> adaptive;

  cnv(size_t cnt, size_t n, size_t m, ptrdiff_t p, ptrdiff_t s, size<tensor> input_sizes) :
          kernels(cnt, tensor(input_sizes.depth, matrix(n, segment(m)))),
          p{p}, s{s} {
    in_size = input_sizes;
    out_size.depth = cnt;
    out_size.height = (in_size.height + 2 * p - n) / s + 1;
    out_size.width = (in_size.width + 2 * p - m) / s + 1;
    for (auto &kernel: kernels) {
      for (auto &layer : kernel) {
        for (auto &row : layer) {
          for (auto &item : row) {
            int x;
            std::cin >> x;
            item = x;
          }
        }
      }
    }
    moments.assign(cnt, tensor(in_size.depth, matrix(n, segment(m, 0))));
    adaptive = moments;
  }


  virtual matrix get_padded_layer(const matrix &layer) = 0;

  virtual void compress_padded_diff_layer(matrix &layer) = 0;

  tensor compute(const tensor &input) final {
    ptrdiff_t d = in_size.depth;
//  ptrdiff_t n = in_size.height;
//  ptrdiff_t m = in_size.width;
    ptrdiff_t k1 = kernels.at(0).at(0).size();
    ptrdiff_t k2 = kernels.at(0).at(0).at(0).size();

    for (ptrdiff_t i = 0; i < d; ++i) {
      padded_input.push_back(get_padded_layer(input.at(i)));
    }
    auto value = tensor(out_size.depth, matrix(out_size.height, segment(out_size.width, 0)));

    for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
      for (ptrdiff_t cur_d = 0; cur_d < d; ++cur_d) {
        for (ptrdiff_t j = 0; j < out_size.height; ++j) {
          for (ptrdiff_t k = 0; k < out_size.width; ++k) {
            // here we have one cell of output
            for (ptrdiff_t dj = 0; dj < k1; ++dj) {
              for (ptrdiff_t dk = 0; dk < k2; ++dk) {
                value.at(i).at(j).at(k) +=
                        padded_input.at(cur_d).at(j * s + dj).at(k * s + dk) * kernels.at(i).at(cur_d).at(dj).at(dk);
              }
            }
          }
        }
      }
    }
    dkernels.assign(kernels.size(), tensor(d, matrix(k1, segment(k2, 0))));
    return value;
  }

  tensor do_backprop(const tensor &output_diff) final {
    auto padded_input_diff = get_padded_backprop(output_diff);
    ptrdiff_t d = in_size.depth;
    ptrdiff_t n = in_size.height;
    ptrdiff_t m = in_size.width;
    tensor res(d, matrix(n, segment(m, 0)));
    for (ptrdiff_t i = 0; i < d; ++i) {
      compress_padded_diff_layer(padded_input_diff.at(i));
      for (ptrdiff_t j = 0; j < n; ++j) {
        for (ptrdiff_t k = 0; k < m; ++k) {
          res.at(i).at(j).at(k) += padded_input_diff.at(i).at(j + p).at(k + p);
        }
      }
    }
    return res;
  }

  tensor get_padded_backprop(const tensor &output_diff) {
    ptrdiff_t d = in_size.depth;
    ptrdiff_t n = in_size.height;
    ptrdiff_t m = in_size.width;
    ptrdiff_t k1 = kernels.at(0).at(0).size();
    ptrdiff_t k2 = kernels.at(0).at(0).at(0).size();
    tensor padded_input_diff(d, matrix(n + 2 * p, segment(m + 2 * p, 0)));

    for (ptrdiff_t i = 0; i < out_size.depth; ++i) {
      for (ptrdiff_t cur_d = 0; cur_d < d; ++cur_d) {
        for (ptrdiff_t j = 0; j < out_size.height; ++j) {
          for (ptrdiff_t k = 0; k < out_size.width; ++k) {
            for (ptrdiff_t dj = 0; dj < k1; ++dj) {
              for (ptrdiff_t dk = 0; dk < k2; ++dk) {
                padded_input_diff.at(cur_d).at(j * s + dj).at(k * s + dk) +=
                        kernels.at(i).at(cur_d).at(dj).at(dk) * output_diff.at(i).at(j).at(k);
                dkernels.at(i).at(cur_d).at(dj).at(dk) +=
                        padded_input.at(cur_d).at(j * s + dj).at(k * s + dk) * output_diff.at(i).at(j).at(k);
              }
            }
          }
        }
      }
    }
    return padded_input_diff;
  }

  void print_parameters_diff() final {
    for (auto &kernel: dkernels) {
      for (auto &layer : kernel) {
        for (auto &row : layer) {
          for (auto &item : row) {
            std::cout << item << " ";
          }
        }
      }
    }
    std::cout << std::endl;
  }
};

struct cnve : cnv {
  using cnv::cnv;

  matrix get_padded_layer(const matrix &layer) final {
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

  void compress_padded_diff_layer(matrix &layer) final {
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
};

struct cnvc : cnv {
  using cnv::cnv;

  matrix get_padded_layer(const matrix &layer) final {
    ptrdiff_t n = layer.size();
    ptrdiff_t m = layer.at(0).size();
    matrix result(n + 2 * p, segment(m + 2 * p));
    for (ptrdiff_t i = 0; i < n; ++i) {
      for (ptrdiff_t j = 0; j < m; ++j) {
        result.at(i + p).at(j + p) = layer.at(i).at(j);
      }
    }

    for (ptrdiff_t i = p; i < p + n; ++i) {
      for (ptrdiff_t j = p - 1; j >= 0; --j) {
        result.at(i).at(j) = result.at(i).at(j + m);
      }
      for (ptrdiff_t j = m + p; j < m + 2 * p; ++j) {
        result.at(i).at(j) = result.at(i).at(j - m);
      }
    }
    for (ptrdiff_t i = p - 1; i >= 0; --i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        result.at(i).at(j) = result.at(i + n).at(j);
      }
    }
    for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        result.at(i).at(j) = result.at(i - n).at(j);
      }
    }
    return result;
  }

  void compress_padded_diff_layer(matrix &layer) final {
    ptrdiff_t n = layer.size() - 2 * p;
    ptrdiff_t m = layer.at(0).size() - 2 * p;
    for (long long i = p - 1; i >= 0; --i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        layer.at(i + n).at(j) += layer.at(i).at(j);
      }
    }
    for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        layer.at(i - n).at(j) += layer.at(i).at(j);
      }
    }

    for (ptrdiff_t i = p; i < p + n; ++i) {
      for (long long j = p - 1; j >= 0; --j) {
        layer.at(i).at(j + m) += layer.at(i).at(j);
      }
      for (ptrdiff_t j = m + p; j < m + 2 * p; ++j) {
        layer.at(i).at(j - m) += layer.at(i).at(j);
      }
    }
  }
};

struct cnvm : cnv {
  using cnv::cnv;

  matrix get_padded_layer(const matrix &layer) final {
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
        result.at(i).at(j) = layer.at(i - p).at(p - j);
      }
      for (ptrdiff_t j = p + m; j < m + 2 * p; ++j) {
        result.at(i).at(j) = layer.at(i - p).at(2 * m + p - j - 2);
      }
    }
    for (ptrdiff_t i = 0; i < p; ++i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        result.at(i).at(j) = result.at(2 * p - i).at(j);
      }
    }
    for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        result.at(i).at(j) = result.at(2 * (n + p) - i - 2).at(j);
      }
    }
    return result;
  }

  void compress_padded_diff_layer(matrix &layer) final {
    ptrdiff_t n = layer.size() - 2 * p;
    ptrdiff_t m = layer.at(0).size() - 2 * p;
    for (ptrdiff_t i = 0; i < p; ++i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        layer.at(2 * p - i).at(j) += layer.at(i).at(j);
      }
    }
    for (ptrdiff_t i = n + p; i < n + 2 * p; ++i) {
      for (ptrdiff_t j = 0; j < m + 2 * p; ++j) {
        layer.at(2 * (n + p) - i - 2).at(j) += layer.at(i).at(j);
      }
    }
    for (ptrdiff_t i = p; i < p + n; ++i) {
      for (ptrdiff_t j = 0; j < p; ++j) {
        layer.at(i).at(p - j + p) += layer.at(i).at(j);
      }
      for (ptrdiff_t j = p + m; j < m + 2 * p; ++j) {
        layer.at(i).at(2 * m + p - j - 2 + p) += layer.at(i).at(j);
      }
    }
  }
};

struct network {
  size<tensor> input_size;
  std::vector<std::unique_ptr<node<tensor, tensor>>> nodes;

  explicit network(size<tensor> sizes) : input_size{sizes} {}

  size<tensor> get_cur_sizes() const {
    if (nodes.empty()) {
      return input_size;
    } else {
      return nodes.back()->out_size;
    }
  }

  void add_bias() {
    nodes.push_back(std::make_unique<bias>(get_cur_sizes()));
  }

  void add_relu(double inv_alpha) {
    nodes.push_back(std::make_unique<relu>(inv_alpha, get_cur_sizes()));
  }

  void add_pool(ptrdiff_t s) {
    nodes.push_back(std::make_unique<pool>(s, get_cur_sizes()));
  }

  void add_cnv(size_t cnt, size_t n, size_t m, ptrdiff_t p, ptrdiff_t s, CNV_TYPE padding) {
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
};

int main() {
  ptrdiff_t n, d;
  std::cin >> n >> d;
  tensor value(d, matrix(n, segment(n)));
  for (ptrdiff_t i = 0; i < d; ++i) {
    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t k = 0; k < n; ++k) {
        int x;
        std::cin >> x;
        value.at(i).at(j).at(k) = x;
      }
    }
  }
  network net({d, n, n});
  ptrdiff_t L;
  std::cin >> L;
  for (ptrdiff_t l = 0; l < L; ++l) {
    std::string type;
    std::cin >> type;
    if (type == "relu") {
      int inv_alpha;
      std::cin >> inv_alpha;
      net.add_relu(inv_alpha);
    } else if (type == "pool") {
      ptrdiff_t s;
      std::cin >> s;
      net.add_pool(s);
    } else if (type == "bias") {
      net.add_bias();
    } else {
      ptrdiff_t h, k, s, p;
      std::cin >> h >> k >> s >> p;
      auto cnv_type = CNV_TYPE::MIRROR;
      if (type == "cnvm") {
        cnv_type = CNV_TYPE::MIRROR;
      } else if (type == "cnve") {
        cnv_type = CNV_TYPE::EXTEND;
      } else {
        cnv_type = CNV_TYPE::CYCLIC;
      }
      net.add_cnv(h, k, k, p, s, cnv_type);
    }
  }
  for (auto &node : net.nodes) {
    value = node->compute(value);
  }
  std::cout << std::fixed;
  print_tensor(value);

  auto n1 = value.size();
  auto n2 = value.at(0).size();
  auto n3 = value.at(0).at(0).size();
  tensor diff(n1, matrix(n2, segment(n3)));
  for (auto &layer : diff) {
    for (auto &row : layer) {
      for (auto &item : row) {
        int x;
        std::cin >> x;
        item = x;
      }
    }
  }
  for (auto it = net.nodes.rbegin(); it != net.nodes.rend(); ++it) {
    diff = it->get()->backprop(diff);
  }
  print_tensor(diff);

  for (auto &item : net.nodes) {
    item->print_parameters_diff();
  }
  return 0;
}
