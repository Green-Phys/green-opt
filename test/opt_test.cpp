/*
 * Copyright (c) 2024 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <green/opt/diis_alg.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <utility>

#include "vector_space.h"

class linear_iterative_solver {
public:
  using X = std::vector<std::complex<double>>;

       linear_iterative_solver(X A, X B) : _A(std::move(A)), _B(std::move(B)) {}

  void solve(const X& x_prev, X& x_new) {
    x_new[0] = (_B[0] - _A[0 * 3 + 1] * x_prev[1] - _A[0 * 3 + 2] * x_prev[2]) / _A[0 * 3 + 0];
    x_new[1] = (_B[1] - _A[1 * 3 + 0] * x_prev[0] - _A[1 * 3 + 2] * x_prev[2]) / _A[1 * 3 + 1];
    x_new[2] = (_B[2] - _A[2 * 3 + 0] * x_prev[0] - _A[2 * 3 + 1] * x_prev[1]) / _A[2 * 3 + 2];
  }

private:
  X _A;
  X _B;
};

TEST_CASE("DIIS") {
  using Vector                   = std::vector<std::complex<double>>;
  using VS                       = green::opt::VSpace;
  using problem_type             = green::opt::optimization_problem<VS::Vector>;
  Vector                       A = {5, -1, -1, -1, 5, 1, -1, 1, 5};
  Vector                       B = {1, 1, 1};
  linear_iterative_solver      solver(A, B);
  green::opt::diis_alg<Vector> diis(2, 5);
  VS                           x_vsp;
  VS                           res_vsp;
  problem_type                 problem;
  auto                         residual = [](Vector& res, VS& x_vsp, problem_type& problem) -> bool {
    if (x_vsp.size() >= 2) {
      Vector last;
      x_vsp.get(x_vsp.size() - 1, last);
      green::opt::add(res, problem.x(), last, std::complex<double>(-1.0, 0.0));  // vec - x_vsp.get_vec(vsp.size()-1);
      return true;
    }
    return false;
  };
  // green::opt::diis_residual<VS::Vector>();
  Vector vec_0{0.5, 1, 0.5};
  Vector solution{0.28571428571428571429, 0.21428571428571428571, 0.21428571428571428571};
  SECTION("TEST C1") {
    Vector vec_prev = vec_0;
    Vector vec_new{0, 0, 0};
    int N_iter = 200;
    for (int i = 0; i < N_iter; ++i) {
      solver.solve(vec_prev, vec_new);
      diis.next_step(vec_new, x_vsp, res_vsp, residual, problem);
      vec_prev = problem.x();
    }
    REQUIRE(std::equal(problem.x().begin(), problem.x().end(), solution.begin(),
                       [](const std::complex<double>& x, const std::complex<double>& s) { return std::abs(x - s) < 1e-7;}));
  }
  SECTION("TEST C2") {
    Vector vec_prev = vec_0;
    Vector vec_new{0, 0, 0};
    int N_iter = 200;
    for (int i = 0; i < N_iter; ++i) {
      solver.solve(vec_prev, vec_new);
      diis.next_step(vec_new, x_vsp, res_vsp, residual, problem, green::opt::lagrangian_type::C2);
      vec_prev = problem.x();
    }
    REQUIRE(std::equal(problem.x().begin(), problem.x().end(), solution.begin(),
                       [](const std::complex<double>& x, const std::complex<double>& s) { return std::abs(x - s) < 1e-7;}));
  }


}