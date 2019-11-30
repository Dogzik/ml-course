#include <memory>
#include <optional>
#include <set>
#include <functional>
#include <iostream>
#include <utility>
#include <cmath>
#include <variant>
#include <cassert>

using namespace std;

int k;

struct node;

struct leaf_node {
  int group;
};

struct tree_node {
  int feature_idx;
  double threshhold;
  unique_ptr<node> left;
  unique_ptr<node> right;
};

struct node {
  int id;
  variant<leaf_node, tree_node> data;
};

struct tree {
  bool use_gini;
  int id;
  using sample_t = vector<std::pair<vector<int>, int>>;
  unique_ptr<node> root;
  int max_depth;
  using part_scorer_t = function<double(sample_t::iterator, sample_t::iterator, sample_t::iterator)>;
  part_scorer_t partition_score;

  explicit tree(int max_depth, part_scorer_t scorer) : max_depth{max_depth}, partition_score{std::move(scorer)} {}

  static int find_most_frequent_klas(sample_t::iterator first, sample_t::iterator last) {
    static vector<int> cnts(k + 1);
    cnts.assign(k + 1, 0);
    for (auto it = first; it != last; ++it) {
      cnts[it->second] += 1;
    }
    return max_element(cnts.begin() + 1, cnts.end()) - cnts.begin();
  }

  static double calc_entropy(std::vector<int> &cnts, int sz) {
    double sum = 0;

    for (auto x: cnts) {
      if (x != 0) {
        sum -= (x * 1.0 / sz) * log(x * 1.0 / sz);
      }
    }
    return sum;
  }


  optional<pair<int, double>> split_entropy(sample_t::iterator first, sample_t::iterator last, int features_cnt) {
    std::optional<double> best_score;
    std::optional<std::pair<int, double>> best_splitter;
    for (int i = 0; i < features_cnt; i++) {
      sort(first, last, [i](sample_t::const_reference a, sample_t::const_reference b) {
        return a.first[i] < b.first[i];
      });
      if (first->first[i] == (last - 1)->first[i]) {
        continue;
      }
      std::vector<int> l_cnt(k + 1);
      std::vector<int> r_cnt(k + 1);
      int l_sz = 0;
      int r_sz = distance(first, last);

      for (auto it = first; it != last; ++it) {
        r_cnt[it->second] += 1;
      }

      int prev = -1;
      for (auto mid = first; mid != last; ++mid) {
        if (mid != first && mid->first[i] != prev) {
          double score = calc_entropy(l_cnt, l_sz) * l_sz + calc_entropy(r_cnt, r_sz) * r_sz;
          if (!best_score.has_value() || (score < *best_score)) {
            best_splitter = std::pair<int, double>{i, (prev + mid->first[i]) / 2.0};
            best_score = score;
          }
        }
        r_cnt[mid->second] -= 1;
        l_cnt[mid->second] += 1;
        ++l_sz;
        --r_sz;
        prev = mid->first[i];
      }
    }
    return best_splitter;
  }

  optional<pair<int, double>> split_gini(sample_t::iterator first, sample_t::iterator last, int features_cnt) {
    std::optional<double> best_score;
    std::optional<std::pair<int, double>> best_splitter;
    for (int i = 0; i < features_cnt; i++) {
      sort(first, last, [i](sample_t::const_reference a, sample_t::const_reference b) {
        return a.first[i] < b.first[i];
      });
      if (first->first[i] == (last - 1)->first[i]) {
        continue;
      }
      std::vector<int> l_cnt(k + 1);
      std::vector<int> r_cnt(k + 1);
      int l_sz = 0;
      int r_sz = distance(first, last);
      long long l_sum = 0;
      long long r_sum = 0;

      auto change = [](int &val, long long &sum, int delta) {
        sum -= val * val;
        val += delta;
        sum += val * val;
      };
      for (auto it = first; it != last; ++it) {
        change(r_cnt[it->second], r_sum, +1);
      }

      int prev = -1;
      for (auto mid = first; mid != last; ++mid) {
        if (mid != first && mid->first[i] != prev) {
          double score = 1.0 * l_sum / l_sz + 1.0 * r_sum / r_sz;
          if (score > best_score) {
            best_splitter = std::pair<int, double>{i, (prev + mid->first[i]) / 2.0};
            best_score = score;
          }
        }
        change(r_cnt[mid->second], r_sum, -1);
        change(l_cnt[mid->second], l_sum, +1);
        ++l_sz;
        --r_sz;
        prev = mid->first[i];
      }
    }
    return best_splitter;
  }

  void train_node(node &cur_node, int cur_depth, sample_t::iterator first, sample_t::iterator last) {
    cur_node.id = ++id;
    int features_cnt = first->first.size();
    auto min_group = min_element(first, last, [](sample_t::const_reference a, sample_t::const_reference b) {
      return a.second < b.second;
    });
    auto max_group = max_element(first, last, [](sample_t::const_reference a, sample_t::const_reference b) {
      return a.second < b.second;
    });
    if (min_group->second == max_group->second) {
      cur_node.data = leaf_node{min_group->second};
      return;
    }
    if (cur_depth == max_depth) {
      cur_node.data = leaf_node{find_most_frequent_klas(first, last)};
      return;
    }
    std::optional<std::pair<int, double>> best_splitter = use_gini ? split_gini(first, last, features_cnt)
                                                                   : split_entropy(first, last, features_cnt);
    if (best_splitter.has_value()) {
      tree_node new_node;
      new_node.feature_idx = best_splitter->first;
      new_node.threshhold = best_splitter->second;
      new_node.left = make_unique<node>();
      new_node.right = make_unique<node>();
      auto mid = partition(first, last, [best_splitter](sample_t::const_reference x) {
        return x.first[best_splitter->first] < best_splitter->second;
      });
      train_node(*new_node.left, cur_depth + 1, first, mid);
      train_node(*new_node.right, cur_depth + 1, mid, last);
      cur_node.data = std::move(new_node);
    } else {
      cur_node.data = leaf_node{find_most_frequent_klas(first, last)};
    }
  }

  void fit(sample_t sample) {
    id = 0;
    root = make_unique<node>();
    use_gini = sample.size() >= 200;
    train_node(*root, 0, sample.begin(), sample.end());
  }

  void print() {
    cout << id << "\n";
    print_node(*root);
  }

  static void print_node(node &cur_node) {
    //cout << "ID: " << cur_node.id << ": ";
    if (holds_alternative<leaf_node>(cur_node.data)) {
      cout << "C " << get<leaf_node>(cur_node.data).group << "\n";
    } else {
      auto &cur = get<tree_node>(cur_node.data);
      cout << "Q " << cur.feature_idx + 1 << " " << cur.threshhold << " " << cur.left->id << " "
           << cur.right->id << "\n";
      print_node(*cur.left);
      print_node(*cur.right);
    }
  }
};

double ginie_score(vector<pair<vector<int>, int>>::iterator first, vector<pair<vector<int>, int>>::iterator mid,
                   vector<pair<vector<int>, int>>::iterator last) {
  static std::vector<int> l_cnt(k + 1);
  static std::vector<int> r_cnt(k + 1);
  l_cnt.assign(k + 1, 0);
  r_cnt.assign(k + 1, 0);
  int l_sum = 0;
  int r_sum = 0;
  for (auto it = first; it != mid; ++it) {
    l_cnt[it->second] += 1;
    ++l_sum;
  }
  for (auto it = mid; it != last; ++it) {
    r_cnt[it->second] += 1;
    ++r_sum;
  }
  long long l = 0;
  long long r = 0;
  for (auto cnt: l_cnt) {
    l += cnt * cnt;
  }
  for (auto cnt: r_cnt) {
    r += cnt * cnt;
  }
  return 1.0 * l / l_sum + 1.0 * r / r_sum;
}


int main() {
  int m, h;
  cin >> m >> k >> h;
  int n;
  cin >> n;
  vector<pair<vector<int>, int>> sample(n, {vector<int>(m), -1});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      cin >> sample[i].first[j];
    }
    cin >> sample[i].second;
  }
  tree classifier(h, ginie_score);
  classifier.fit(std::move(sample));
  cout << fixed;
  cout.precision(12);
  classifier.print();
  return 0;
}
