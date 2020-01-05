#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

vector<int> get_rangs(const vector<int> &arr) {
  if (arr.empty()) {
    return {};
  }
  vector<pair<int, int>> indexed_arr(arr.size());
  for (int i = 0; i < arr.size(); ++i) {
    indexed_arr[i] = {arr[i], i};
  }
  sort(indexed_arr.begin(), indexed_arr.end());
  vector<int> res(arr.size());
  res[indexed_arr[0].second] = 0;
  int cur_rang = 0;
  for (int i = 1; i < indexed_arr.size(); ++i) {
    if (indexed_arr[i - 1].first != indexed_arr[i].first) {
      ++cur_rang;
    }
    res[indexed_arr[i].second] = cur_rang;
  }
  return res;
}

double spirman_coff(const vector<int> &a, const vector<int> &b) {
  int n = a.size();
  if (n == 0 || n == 1) {
    return 0;
  }
  auto a_rangs = get_rangs(a);
  auto b_rangs = get_rangs(b);
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    long long d = a_rangs[i] - b_rangs[i];
    sum += d * d;
  }
  return 1 - 6 * sum / ((n - 1.0) * n * (n + 1.0));
}

int main() {
  int n;
  cin >> n;
  vector<int> x(n);
  vector<int> y(n);
  for (int i = 0; i < n; ++i) {
    cin >> x[i] >> y[i];
  }
  cout << spirman_coff(x, y);
}
