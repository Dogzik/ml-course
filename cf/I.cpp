#include <iostream>

using namespace std;

using uint = unsigned int;

int ones_cnt(uint x) {
  int res = 0;
  while (x > 0) {
    ++res;
    x = x & (x - 1);
  }
  return res;
}

int main() {
  uint m;
  cin >> m;
  cout << fixed;
  cout.precision(12);
  cout << 2 << endl;
  cout << (1u << m) << " " << 1 << endl;
  for (uint mask = 0; mask < (1u << m); ++mask) {
    for (uint pos = 0; pos < m; ++pos) {
      if (mask & (1u << pos)) {
        cout << "1.0 ";
      } else {
        cout << "-1000000000.0 ";
      }
    }
    cout << 0.5 - ones_cnt(mask) << endl;
  }
  for (uint i = 0; i < (1u << m); ++i) {
    int x;
    cin >> x;
    cout << static_cast<double>(x) << " ";
  }
  cout << "-0.5" << endl;
  return 0;
}
