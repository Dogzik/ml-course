#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>

using namespace std;

long long internal_distance(const vector<pair<int, int>> &samples) {
  unordered_map<int, vector<int>> classes;
  for (auto sample : samples) {
    classes[sample.second].push_back(sample.first);
  }
  for (auto &item : classes) {
    sort(item.second.begin(), item.second.end());
  }
  long long res = 0;
  for (auto &item : classes) {
    auto &xs = item.second;
    long long suff_sum = accumulate(xs.begin(), xs.end(), 0LL);
    long long pref_sum = 0;
    for (long long i = 0; i < xs.size(); ++i) {
      pref_sum += xs[i];
      suff_sum -= xs[i];
      long long pref_sz = i + 1;
      long long suff_sz = xs.size() - pref_sz;
      res += (xs[i] * pref_sz - pref_sum) + (suff_sum - xs[i] * suff_sz);
    }
  }
  return res;
}

long long external_distance(const vector<pair<int, int>> &samples) {
  auto copy = samples;
  unordered_map<int, long long> pref_sums;
  unordered_map<int, long long> suff_sums;
  long long total_pref_sum = 0;
  long long total_suff_sum = 0;
  unordered_map<int, long long> pref_cnt;
  unordered_map<int, long long> suff_cnt;
  sort(copy.begin(), copy.end());
  for (auto elem : copy) {
    suff_sums[elem.second] += elem.first;
    suff_cnt[elem.second] += 1;
    total_suff_sum += elem.first;
  }
  long long res = 0;
  for (long long i = 0; i < copy.size(); ++i) {
    auto elem = copy[i];
    total_pref_sum += elem.first;
    total_suff_sum -= elem.first;
    pref_sums[elem.second] += elem.first;
    suff_sums[elem.second] -= elem.first;
    pref_cnt[elem.second] += 1;
    suff_cnt[elem.second] -= 1;

    long long total_pref_cnt = i + 1;
    long long total_suff_cnt = copy.size() - total_pref_cnt;

    long long good_pref_cnt = total_pref_cnt - pref_cnt[elem.second];
    long long good_suff_cnt = total_suff_cnt - suff_cnt[elem.second];

    long long good_pref_sum = total_pref_sum - pref_sums[elem.second];
    long long good_suff_sum = total_suff_sum - suff_sums[elem.second];

    res += good_pref_cnt * elem.first - good_pref_sum;
    res += good_suff_sum - good_suff_cnt * elem.first;
  }
  return res;
}


int main() {
  unordered_map<int, vector<int>> classes;
  int k, n;
  cin >> k >> n;
  vector<pair<int, int>> samples(n);
  for (int i = 0; i < n; i++) {
    cin >> samples[i].first >> samples[i].second;
  }
  cout << internal_distance(samples) << endl;
  cout << external_distance(samples) << endl;
}

