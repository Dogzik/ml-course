#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace std;

constexpr double EPS = 1e-7;
auto random_gen = mt19937_64(time(nullptr));

struct SVM {
  vector<vector<double>> kernel;
  vector<int> y;
  double C;
  double b;
  vector<double> alpha;
  size_t N;
  size_t MAX_ITERATIONS;

  SVM(vector<vector<double>>&& kernel, vector<int>&& y, double C)
  : kernel(std::move(kernel))
  , y(std::move(y))
  , C{C}
  , b{0.0}
  , alpha(this->y.size(), 0)
  , N{alpha.size()}
  , MAX_ITERATIONS{N * 6000}
  {}

  double calc_E(size_t idx) {
    double res = 0;
    for (size_t i = 0; i < N; ++i) {
      res += y[i] * alpha[i] * kernel[i][idx];
    }
    return res - y[idx];
  }

  size_t get_j(size_t i) {
    auto distribution = uniform_int_distribution<size_t>(0, N - 2);
    size_t j = distribution(random_gen);
    return j < i ? j : j + 1;
  }

  std::pair<double, double> get_U_V(size_t i, size_t j) {
    if (y[i] == y[j]) {
      return {max(0.0, alpha[i] + alpha[j] - C), min(C, alpha[i] + alpha[j])};
    } else {
      return {max(0.0, alpha[j] - alpha[i]), min(C, C + alpha[j] - alpha[i])};
    }
  }

  double get_b(size_t idx) {
    double res = 0;
    for (size_t i = 0; i < N; ++i) {
      res += alpha[i] * y[i] * kernel[i][idx];
    }
    return 1.0 / y[idx] - res;
  }

  void calc_b() {
    int idx = -1;
    for (size_t i = 0; i < N; ++i) {
      if (EPS < alpha[i] && alpha[i] + EPS < C) {
        idx = i;
        break;
      }
    }
    if (idx != -1) {
      b = get_b(idx);
      return;
    }
    size_t cnt = 0;
    for (size_t i = 0; i < N; ++i) {
      if (EPS < alpha[i]) {
        b += get_b(i);
        ++cnt;
      }
    }
    b /= cnt;
  }

  void fit() {
    vector<size_t> indices(N);
    iota(indices.begin(), indices.end(), 0);
    size_t iterations_cnt = 0;
    while (iterations_cnt < MAX_ITERATIONS) {
      shuffle(indices.begin(), indices.end(), random_gen);
      for (size_t i_fake = 0; i_fake < N && iterations_cnt < MAX_ITERATIONS; ++i_fake, ++iterations_cnt) {
        size_t i = indices[i_fake];
        size_t j = indices[get_j(i_fake)];
        auto E_i = calc_E(i);
        auto E_j = calc_E(j);
        auto prev_alpha_i = alpha[i];
        auto prev_alpha_j = alpha[j];
        auto [U, V] = get_U_V(i, j);
        if (V - U < EPS) {
          continue;
        }
        auto eta = 2.0 * kernel[i][j] - (kernel[i][i] + kernel[j][j]);
        if (eta > -EPS) {
          continue;
        }
        auto possible_new_alpha_j = prev_alpha_j + y[j] * (E_j - E_i) / eta;
        auto new_alpha_j = min(max(U, possible_new_alpha_j), V);
        if (abs(new_alpha_j - prev_alpha_j) < EPS) {
          continue;
        }
        alpha[j] = new_alpha_j;
        alpha[i] += y[i] * y[j] * (prev_alpha_j - new_alpha_j);
      }
    }
    calc_b();
  }
};

int main() {
  size_t n;
  cin >> n;
  vector<vector<double>> kernel(n, vector<double>(n));
  vector<int> y(n);
  for (size_t i = 0; i < n; ++i) {
    int x;
    for (size_t j = 0; j < n; ++j) {
      cin >> x;
      kernel[i][j] = x;
    }
    cin >> y[i];
  }
  double C;
  cin >> C;
  SVM cls(std::move(kernel), std::move(y), C);
  cls.fit();
  cout << fixed;
  cout.precision(12);
  for (auto alpha : cls.alpha) {
    cout << alpha << "\n";
  }
  cout << cls.b << "\n";
  return 0;
}
