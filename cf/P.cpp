#include <vector>
#include <iostream>
using namespace std;


/**
 * D(Y|X) = E(Y^2|X) - (E(Y|X))^2
 * Also E(E(A|B)) = E(A)
 * So E(D(Y|X)) = E(Y^2) - E((E(Y|X))^2)
 */
int main() {
  int k, n;
  cin >> k >> n;
  vector<pair<int, int>> elems(n);
  for (int i = 0; i < n; ++i) {
    cin >> elems[i].first >> elems[i].second;
  }
  double second_moment_y = 0;
  for (auto [x, y] : elems) {
    second_moment_y += y / static_cast<double>(n) * y;
  }
  std::vector<pair<double , double>> expectation_y_from_x(k, {0, 0});
  for (auto [x, y] : elems) {
    expectation_y_from_x[x - 1].second += 1.0 / n; // P(X_i)
    expectation_y_from_x[x - 1].first += y / static_cast<double>(n); // E(Y, X_i)
  }
  double expectation_of_conditional_square = 0;
  for (auto [e_y, p] : expectation_y_from_x) {
    if (p != 0) {
      expectation_of_conditional_square += e_y * e_y / p;
    }
  }
  cout << fixed << second_moment_y - expectation_of_conditional_square << endl;
  return 0;
}
