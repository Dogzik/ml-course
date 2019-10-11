#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <ctime>
#include <chrono>
using namespace std;

struct point {
  vector<double> x;
  double y;

  explicit point(size_t sz) : x(sz), y{0} {
    x.back() = 1;
  };
};

size_t n, m;

double calc_scalar_product(vector<double> const &a, vector<double> const &b) {
  double res = 0;
  for (size_t i = 0; i < m; ++i) {
    res += a[i] * b[i];
  }
  return res;
}

vector<double> calc_grad(point const points[], size_t cnt, const vector<double> &w) {
  vector<double> diffs(cnt);
  for (size_t i = 0; i < cnt; ++i) {
    diffs[i] = 2 * (points[i].y - calc_scalar_product(w, points[i].x));
  }
  vector<double> grad(m, 0);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < cnt; ++j) {
      grad[i] -= diffs[j] * points[j].x[i];
    }
    grad[i] /= cnt;
  }
  return grad;
}

vector<double> calc_step(point const points[], size_t cnt, const vector<double> &w) {
  vector<double> diffs(cnt);
  for (size_t i = 0; i < cnt; ++i) {
    diffs[i] = (points[i].y - calc_scalar_product(w, points[i].x));
  }
  vector<double> grad(m, 0);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < cnt; ++j) {
      grad[i] -= 2 * diffs[j] * points[j].x[i];
    }
    grad[i] /= cnt;
  }

  vector<double> scalar_products(cnt, 0);
  for (size_t i = 0; i < cnt; ++i) {
    scalar_products[i] = calc_scalar_product(grad, points[i].x);
  }
  double b = 0;
  for (size_t i = 0; i < cnt; ++i) {
    b += scalar_products[i] * scalar_products[i];
  }
  double mu = 0;
  if (b == 0) {
    mu = 0;
  } else {
    for (size_t i = 0; i < cnt; ++i) {
      mu += diffs[i] * scalar_products[i];
    }
    mu = -mu / b;
  }
  for (size_t i = 0; i < m; ++i) {
    grad[i] *= mu;
  }
  return grad;
}

int main() {
  auto start = chrono::system_clock::now();
  cin >> n >> m;
  ++m;
  vector<point> points(n, point(m));
  for (size_t i = 0; i < n; ++i) {
    int tmp;
    for (size_t j = 0; j < m - 1; ++j) {
      cin >> tmp;
      points[i].x[j] = tmp;
    }
    cin >> tmp;
    points[i].y = tmp;
  }
  vector<double> w(m, 0);
  if (m == 2) {
    double sum_x0 = 0;
    double sum_y = 0;
    for (auto &dot : points) {
      sum_x0 += dot.x[0];
      sum_y += dot.y;
    }
    double avg_x0 = sum_x0 / n;
    double avg_y = sum_y / n;
    double a = 0;
    double b = 0;
    for (auto &dot : points) {
      b += (dot.x[0] - avg_x0) * (dot.x[0] - avg_x0);
      a += (dot.x[0] - avg_x0) * (dot.y - avg_y);
    }
    w[0] = (b == 0) ? 0 : a / b;
    w[1] = avg_y - w[0] * avg_x0;
  } else {
    size_t batch_size = min(size_t{1}, n);
    auto random_engine = mt19937_64(4);
    bool stop = false;
    while (!stop) {
      shuffle(points.begin(), points.end(), random_engine);
      size_t j;
      for (j = 0; j < n - batch_size + 1; j += batch_size) {
        auto step = calc_step(&points[j], batch_size, w);
        for (size_t k = 0; k < m; ++k) {
          w[k] -= step[k];
        }
        if (chrono::system_clock::now() - start >= chrono::milliseconds(750)) {
          stop = true;
          break;
        }
      }
      if (j != n - batch_size + 1) {
        j -= batch_size;
        auto cnt = n - j;
        auto step = calc_step(&points[j], cnt, w);
        for (size_t k = 0; k < m; ++k) {
          w[k] -= step[k];
        }
      }
    }
  }
  for (auto x : w) {
    cout.precision(std::numeric_limits<double>::max_digits10);
    cout << x << "\n";
  }
  return 0;
}
