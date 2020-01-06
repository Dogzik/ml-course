#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
using namespace std;

namespace std {
  template<>
  struct hash<pair<int, int>> {
    size_t operator()(const pair<int, int> &elem) const noexcept {
      size_t hash_first = hash<int>{}(elem.first);
      size_t hash_second = hash<int>{}(elem.second);
      const int64_t m = 0xc6a4a7935bd1e995;
      const int32_t r = 47;

      hash_second *= m;
      hash_second ^= hash_second >> r;
      hash_second *= m;

      hash_first ^= hash_second;
      hash_first *= m;
      hash_first += 0xe6546b64;
      return hash_first;
    }
  };
}

int main() {
  int kx, ky;
  cin >> kx >> ky;
  int n;
  cin >> n;
  vector<pair<int, int>> elems(n);
  unordered_map<int, double> p_x;
  unordered_map<pair<int, int>, double> p_xy;
  for (int i = 0; i < n; ++i) {
    cin >> elems[i].first >> elems[i].second;
    p_x[elems[i].first] += 1.0 / n;
    p_xy[{elems[i].first, elems[i].second}] += 1.0 / n;
  }
  double res = 0;
  for (auto [x_y, cur_p_xy] : p_xy) {
    auto x = x_y.first;
    res += -cur_p_xy * (log(cur_p_xy) - log(p_x.at(x)));
  }
  cout << fixed;
  cout.precision(10);
  cout << res << endl;
}
