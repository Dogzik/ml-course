#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <unordered_map>

using namespace std;

using uint = uint32_t;

namespace std {
  template<>
  struct hash<pair<uint, uint>> {
    size_t operator()(const pair<uint, uint> &elem) const noexcept {
      size_t hash_first = hash<uint>{}(elem.first);
      size_t hash_second = hash<uint>{}(elem.second);
      const uint64_t m = 0xc6a4a7935bd1e995;
      const uint32_t r = 47;

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
  uint k1, k2;
  cin >> k1 >> k2;
  uint n;
  cin >> n;
  vector<uint> x1(n), x2(n);
  vector<double> f1(k1, 0), f2(k2, 0);
  unordered_map<pair<uint, uint>, uint> cross_cnt;
  for (uint i = 0; i < n; ++i) {
    cin >> x1[i] >> x2[i];
    x1[i] -= 1;
    x2[i] -= 1;
    f1[x1[i]] += 1;
    f2[x2[i]] += 1;
    cross_cnt[{x1[i], x2[i]}] += 1;
  }
  for (auto &x : f1) {
    x /= n;
  }
  for (auto &x : f2) {
    x /= n;
  }
  double res = n;
  for (auto[values, real] : cross_cnt) {
    auto[a, b] = values;
    double expected = static_cast<double>(n) * f1[a] * f2[b];
    double diff = real - expected;
    res -= expected;
    res += diff * diff / expected;
  }
  cout << fixed << res << endl;
  return 0;
}
