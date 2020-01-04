
#include <iostream>
#include <vector>
#include <optional>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <thread>

using namespace std;


struct node {
  vector<node *> inputs;
  vector<vector<double>> value;
  vector<vector<double>> diff;

  virtual void compute() = 0;

  virtual void spread_diff() = 0;

  void initialize_diff() {
    diff.assign(value.size(), vector<double>(value[0].size(), 0));
  }

  void read_diff() {
    for (auto &row : diff) {
      for (auto &cell : row) {
        int x;
        cin >> x;
        cell = x;
      }
    }
  }

  explicit node(vector<node *> inputs) : inputs{move(inputs)}, value{} {}
};

struct var : node {
  var() : node(vector<node *>{}) {}

  void set_data(vector<vector<double>> data) {
    value = move(data);
  }

  void compute() final {
    initialize_diff();
  }

  void spread_diff() final {}
};

struct tnh : node {
  explicit tnh(node *source) : node(vector<node *>{source}) {}

  void compute() final {
    //assert(inputs.size() == 1);
//    inputs.front()->compute();
    value = inputs.front()->value;
    for (auto &row : value) {
      for (auto &cell : row) {
        cell = tanh(cell);
      }
    }
    initialize_diff();
  }

  void spread_diff() final {
    for (size_t i = 0; i < value.size(); ++i) {
      for (size_t j = 0; j < value[0].size(); ++j) {
        auto cur_value = value[i][j];
        inputs.front()->diff[i][j] += (1 - cur_value * cur_value) * diff[i][j];
      }
    }
  }
};

struct rlu : node {
  double inv_alpha;

  rlu(double inv_alpha, node *source) : node(vector<node *>{source}), inv_alpha{inv_alpha} {}

  void compute() final {
    //assert(inputs.size() == 1);
//    inputs.front()->compute();
    value = inputs.front()->value;
    for (auto &row : value) {
      for (auto &cell : row) {
        if (cell < 0) {
          cell /= inv_alpha;
        }
      }
    }
    initialize_diff();
  }

  void spread_diff() final {
    for (size_t i = 0; i < value.size(); ++i) {
      for (size_t j = 0; j < value[0].size(); ++j) {
        auto cur_input = inputs.front()->value[i][j];
        double multiplier;
        if (cur_input >= 0) {
          multiplier = 1.0;
        } else {
          multiplier = 1.0 / inv_alpha;
        }
        inputs.front()->diff[i][j] += multiplier * diff[i][j];
      }
    }
  }
};

struct mul : node {
  mul(node *a, node *b) : node(vector<node *>{a, b}) {}

  void compute() final {
    //assert(inputs.size() == 2);
//    inputs[0]->compute();
//    inputs[1]->compute();
    auto &a = inputs[0]->value;
    auto &b = inputs[1]->value;
    auto n = a.size();
    auto m = a[0].size();
    //assert(m == b.size());
    auto k = b[0].size();
    value = vector<vector<double>>(n, vector<double>(k, 0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        for (size_t t = 0; t < m; ++t) {
          value[i][j] += a[i][t] * b[t][j];
        }
      }
    }
    initialize_diff();
  }

  void spread_diff() final {
    auto &a = inputs[0]->value;
    auto &b = inputs[1]->value;
    auto &da = inputs[0]->diff;
    auto &db = inputs[1]->diff;
    auto n = a.size();
    auto m = a[0].size();
    auto k = b[0].size();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        double cur_diff = 0;
        for (size_t t = 0; t < k; ++t) {
          cur_diff += diff[i][t] * b[j][t];
        }
        da[i][j] += cur_diff;
      }
    }
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        double cur_diff = 0;
        for (size_t t = 0; t < n; ++t) {
          cur_diff += a[t][i] * diff[t][j];
        }
        db[i][j] += cur_diff;
      }
    }
  }
};

struct sum : node {
  using node::node;

  void compute() final {
    //assert(!inputs.empty());
//    inputs.front()->compute();
    auto n = inputs.front()->value.size();
    auto m = inputs.front()->value[0].size();
    //assert(n > 0 && m > 0);
    value = vector<vector<double>>(n, vector<double>(m, 0));
    for (auto input : inputs) {
//      input->compute();
      auto &cur = input->value;
      //assert(cur.size() == n);
      //assert(cur[0].size() == m);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
          value[i][j] += cur[i][j];
        }
      }
    }
    initialize_diff();
  }

  void spread_diff() final {
    for (size_t i = 0; i < value.size(); ++i) {
      for (size_t j = 0; j < value[0].size(); ++j) {
        for (auto input : inputs) {
          input->diff[i][j] += diff[i][j];
        }
      }
    }
  }
};

struct had : node {
  using node::node;

  void compute() final {
    //assert(!inputs.empty());
//    inputs.front()->compute();
    auto n = inputs.front()->value.size();
    auto m = inputs.front()->value[0].size();
    //assert(n > 0 && m > 0);
    value = vector<vector<double>>(n, vector<double>(m, 1));
    for (auto input : inputs) {
//      input->compute();
      auto &cur = input->value;
      //assert(cur.size() == n);
      //assert(cur[0].size() == m);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
          value[i][j] *= cur[i][j];
        }
      }
    }
    initialize_diff();
  }

  void spread_diff() final {
    for (size_t i = 0; i < value.size(); ++i) {
      for (size_t j = 0; j < value[0].size(); ++j) {
        for (size_t k = 0; k < inputs.size(); ++k) {
          double multiplier = 1;
          for (size_t t = 0; t < inputs.size(); ++t) {
            if (t != k) {
              multiplier *= inputs[t]->value[i][j];
            }
          }
          inputs[k]->diff[i][j] += multiplier * diff[i][j];
        }
      }
    }
  }
};

struct network {
  vector<unique_ptr<node>> nodes;

  [[nodiscard]] vector<node *> get_inputs(vector<size_t> const &inputs_idxs) const {
    vector<node *> inputs;
    inputs.reserve(inputs_idxs.size());
    for (auto idx : inputs_idxs) {
      inputs.push_back(nodes[idx].get());
    }
    return inputs;
  }

  void add_node(node *new_node) {
    nodes.emplace_back(new_node);
  }

  void print_node(size_t idx) {
//    nodes[idx]->compute();
    for (auto &row : nodes[idx]->value) {
      for (auto cell : row) {
        cout << cell << " ";
      }
      cout << "\n";
    }
  }

  void print_diff(size_t idx) {
    for (auto &row : nodes[idx]->diff) {
      for (auto cell : row) {
        cout << cell << " ";
      }
      cout << "\n";
    }
  }

  void compute() {
    for (auto &ptr : nodes) {
      ptr->compute();
    }
  }

  void backprop() {
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      it->get()->spread_diff();
    }
  }
};

int main() {
  size_t n, m, k;
  cin >> n >> m >> k;
  vector<pair<size_t, size_t>> inputs_sizes;
  inputs_sizes.reserve(m);
  network net;
  for (size_t i = 0; i < n; ++i) {
    string type;
    cin >> type;
    if (type == "var") {
      size_t a, b;
      cin >> a >> b;
      inputs_sizes.emplace_back(a, b);
      net.add_node(new var{});
    } else if (type == "tnh") {
      size_t x;
      cin >> x;
      net.add_node(new tnh(net.nodes[x - 1].get()));
    } else if (type == "rlu") {
      int inv_alpha;
      size_t x;
      cin >> inv_alpha >> x;
      net.add_node(new rlu(inv_alpha, net.nodes[x - 1].get()));
    } else if (type == "mul") {
      size_t a, b;
      cin >> a >> b;
      net.add_node(new mul(net.nodes[a - 1].get(), net.nodes[b - 1].get()));
    } else if (type == "sum") {
      size_t len;
      cin >> len;
      vector<size_t> idxs(len);
      for (size_t j = 0; j < len; ++j) {
        cin >> idxs[j];
        --idxs[j];
      }
      auto inputs = net.get_inputs(idxs);
      net.add_node(new sum(move(inputs)));
    } else if (type == "had") {
      size_t len;
      cin >> len;
      vector<size_t> idxs(len);
      for (size_t j = 0; j < len; ++j) {
        cin >> idxs[j];
        --idxs[j];
      }
      auto inputs = net.get_inputs(idxs);
      net.add_node(new had(move(inputs)));
    } else {
      //assert(!"Wrong input type");
    }
  }
  for (size_t i = 0; i < m; ++i) {
    vector<vector<double>> input(inputs_sizes[i].first, vector<double>(inputs_sizes[i].second));
    for (size_t a = 0; a < inputs_sizes[i].first; ++a) {
      for (size_t b = 0; b < inputs_sizes[i].second; ++b) {
        int x;
        cin >> x;
        input[a][b] = x;
      }
    }
    dynamic_cast<var *>(net.nodes[i].get())->set_data(move(input));
  }
  net.compute();
  cout << fixed;
  cout.precision(12);
  for (size_t i = n - k; i < n; ++i) {
    net.print_node(i);
  }
  for (size_t i = n - k; i < n; ++i) {
    net.nodes[i]->read_diff();
  }
  net.backprop();
  for (size_t i = 0; i < m; ++i) {
    net.print_diff(i);
  }
  return 0;
}
