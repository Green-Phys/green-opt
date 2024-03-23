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

#ifndef GREEN_OPT_DIIS_ALG
#define GREEN_OPT_DIIS_ALG

#define DIIS_DEBUG 0

#include <green/utils/mpi_shared.h>

#include <Eigen/Dense>
#include <complex>

#include "iterative_optimizer.h"

namespace green::opt {

  enum class lagrangian_type { C1, C2 };

  /**
   * \brief Implementation of the DIIS method
   *
   *  <b> DIIS <b>
   *  The implementation closely follows the original ideas from P. Pulay
   *  and further publications.
   *  The implementation does not use any specific choise of the residual---
   *  it is defined outside of the class and passed as a pointer.
   *  Users can use their own residuals inheriting the existing one.
   *  The most well-tested choice is the definition as a difference between subsequent iterations
   *  \f[ r_i = x_i - x_{i-1} \f]
   *  The space of probe vectors used for extrapolation and residuals are stored
   *  in the corresponding linear spaces.
   *  The internal linear system (composed of residual overlaps and Lagrange multipliers)
   *  is solved in a numerically stable way:
   *  \f[ B  1  * (c) = (0) \\
   *      1  0  * (l)   (1) \f]
   *
   *  Since the B matrix of overlaps of residuals can be very small, such system is badly conditioned.
   *  To avoid numerical instabilities, the system is modified:
   *  \f[ B c = 1 \f]
   *  And then the coefficients are normalized such that the constraint is satisfied.
   *  Both C1 and C2 versions of the Lagrangian are implemented.
   *  Numerically they do not seem to differ at all.
   *  At the moment only C1 version is enabled.
   *
   */
  template <typename Vector>
  class diis_alg : public iterative_optimizer<Vector, diis_alg<Vector>> {
  private:
    using MatrixXcd = Eigen::MatrixXcd;
    using VectorXcd = Eigen::VectorXcd;
    using iterative_optimizer<Vector, diis_alg>::trust_norm;

    MatrixXcd         _m_B;  // Overlap matrix of the residuals
    VectorXcd         _m_C;  // Vector of extrapolation coefs
    size_t            _min_subsp_size;
    size_t            _max_subsp_size;
    const std::string diis_str{"DIIS: "};

  public:
    explicit diis_alg(size_t min_subsp_size = 1, size_t max_subsp_size = 10) :
        _min_subsp_size(min_subsp_size), _max_subsp_size(max_subsp_size) {}

    double get_err_norm() {
      size_t dim = _m_B.cols();
      if (dim == 0) {
        if (!!utils::context.global_rank) std::cout << diis_str << "The error matrix has zero dimension" << std::endl;
        return 1.0;
      }
      return std::sqrt(std::abs(_m_B(dim - 1, dim - 1)));
    }

    void print_err_norm() {
      size_t dim = _m_B.cols();
      if (dim == 0) {
        if (!utils::context.global_rank) std::cout << diis_str << "The error matrix has zero dimension" << std::endl;
      } else {
        if (!utils::context.global_rank) std::cout << diis_str << "The error norm = " << get_err_norm() << std::endl;
      }
    }

    void print_B() {
      std::cout << diis_str << "m_B:" << std::endl;
      std::cout << _m_B << std::endl;
    }

    void print_C() {
      std::cout << diis_str << "Extrapolation coefs: " << std::endl;
      std::cout << _m_C << std::endl;
    }

    void init(size_t max_subsp_size, double trust) {
      _max_subsp_size = max_subsp_size;
      trust_norm()    = trust;
#if DIIS_DEBUG
      if (!!utils::context.global_rank) print_B();
#endif
    };

    template <typename VS, typename Res>
    void next_step(Vector& vec, VS& x_vsp, VS& res_vsp, Res& residual, optimization_problem<Vector>& problem) {
      if (x_vsp.size() <= _min_subsp_size) {
        std::cout << diis_str << "Growing subspace without extrapolation\n";
        x_vsp.add(vec);
        problem.x() = vec;
        return;
      }
      // Normal execution
      if (res_vsp.size() == _max_subsp_size) {
        if (!utils::context.global_rank)
          std::cout << diis_str << "Reached maximum subspace. The first vector will be kicked out of the subspace." << std::endl;
        res_vsp.purge(0);  // can do it smarter and purge the one with the smallest coef
        x_vsp.purge(0);
        purge_overlap(0);
      }
      problem.x() = vec;
      Vector res;
      if (!residual(res, x_vsp, problem)) {
        if (!utils::context.global_rank) std::cout << diis_str << "Could not get residual!!! ABORT!" << std::endl;
      }
      // TODO: treat linear deps
      update_overlaps(res, res_vsp);
      res_vsp.add(res);
      x_vsp.add(vec);
      if (res_vsp.size() > 1) {
        compute_coefs(lagrangian_type::C1);

        if (!utils::context.global_rank) std::cout << diis_str << "Performing the DIIS extrapolation..." << std::endl;
        if (!utils::context.global_rank) print_B();
        if (!utils::context.global_rank)
          std::cout << diis_str << "Predicted extrapol (e,e) = " << _m_C.dot(_m_B * _m_C) << std::endl;
        Vector result = x_vsp.make_linear_comb(_m_C);
        problem.x()   = result;
      } else {
        problem.x() = vec;
      }
    }

  private:
    /**
     * Remove overlaps with the vector k
     *     The dimensions of the matrix m_B are shrinked by 1
     */
    void purge_overlap(const size_t k) {
      MatrixXcd Bnew(_m_B.rows() - 1, _m_B.cols() - 1);
      for (size_t i = 0, mi = 0; i < Bnew.rows(); i++, mi++) {
        if (i == k) ++mi;
        for (size_t j = 0, mj = 0; j < Bnew.cols(); j++, mj++) {
          if (j == k) ++mj;
          Bnew(i, j) = _m_B(mi, mj);
        }
      }
      _m_B = Bnew;
    }

    /**
     * Add overlaps with the new residual vector
     * The dimensions of the matrix m_B are extended by 1
     *
     * @param res - new residual vector
     * @param res_vsp - residual vector space
     */
    template <typename VS>
    void update_overlaps(Vector& res, VS& res_vsp) {
#if DIIS_DEBUG
      if (!!utils::context.global_rank) print_B();
      if (m_B.cols() > 1) {
        std::cout << "Before the update" << std::endl;
        print_B();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(m_B);
        std::cout << "evals: " << es.eigenvalues().transpose() << std::endl;
        std::cout << "updating..." << std::endl;
      }
#endif

      MatrixXcd Bnew = MatrixXcd::Zero(_m_B.rows() + 1, _m_B.cols() + 1);

      // Assing what is known already:
      for (size_t i = 0; i < _m_B.rows(); i++) {
        for (size_t j = 0; j < _m_B.cols(); j++) Bnew(i, j) = _m_B(i, j);
      }

      // Evaluate new overlaps and add them to B:
      // Can ship this piece as a function with the vector space
      // for good parallelization
      for (size_t i = 0; i < _m_B.cols(); i++) {
        Bnew(i, _m_B.cols()) = res_vsp.overlap(i, res);
        Bnew(_m_B.cols(), i) = std::conj(Bnew(i, _m_B.cols()));
      }
      Bnew(_m_B.cols(), _m_B.cols()) = res_vsp.overlap(res, res);
      _m_B                           = Bnew;
#if DIIS_DEBUG
      std::cout << "After the update" << std::endl;
      print_B();
#endif
    }

    /**
     * Compute new Lagrange coefficients
     * @param type type of lagrange coefficients to compute
     */
    void compute_coefs(const lagrangian_type type) {
      switch (type) {
        case lagrangian_type::C1:
          compute_coefs_c1();
          break;
        case lagrangian_type::C2:
          compute_coefs_c2();
          break;
        default:
          compute_coefs_c1();
      }
      if (!utils::context.global_rank) print_cond();
      if (!utils::context.global_rank) print_C();
    }

    /**
     * Compute and print the condition number
     */
    void print_cond() const {
      Eigen::JacobiSVD svd(_m_B);
      double           cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
      if (!utils::context.global_rank)
        std::cout << diis_str << "Condition number of the residual overlap matrix: " << cond << std::endl;
    }

    void compute_coefs_c1() {
#if DIIS_DEBUG
      print_B();
#endif
      MatrixXcd            B  = _m_B.real();
      VectorXcd            bb = VectorXcd::Constant(B.cols(), 1.0);

      Eigen::FullPivLU<MatrixXcd> solver(B);
      VectorXcd            x  = solver.solve(bb);
      std::complex<double> sum(0.0, 0.0);
      for (size_t i = 0; i < B.rows(); i++) {
        sum += x[i];
      }
      _m_C = (x / sum).real();
    }

    void compute_coefs_c2() {
#if DIIS_DEBUG
      print_B();
#endif

#pragma float_control(precise, on)  // Need accurate extrapolation coeffs
      MatrixXcd           B  = _m_B;

      VectorXcd           bb = VectorXcd::Constant(B.cols(), 1.0);

      Eigen::BDCSVD       svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
      MatrixXcd           U         = svd.matrixU();
      VectorXcd           sing_vals = svd.singularValues().asDiagonal();
      std::vector<double> errors(B.cols());

      // Normalize columns such that sum_i c_i = 1
      // and evaluate errors for all solutions
      for (size_t j = 0; j < B.cols(); j++) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t i = 0; i < B.rows(); i++) {
          sum += U(i, j);
        }
        U.col(j) /= sum;
        errors[j] = std::abs(sing_vals(j) * U.col(j).dot(U.col(j)));
      }
      // find the solution minimizing the error
      auto   it           = std::min_element(std::begin(errors), std::end(errors));
      size_t column_index = std::distance(std::begin(errors), it);
      _m_C                = U.col(column_index);
    }
  };

}  // namespace green::opt

#endif  // GREEN_OPT_DIIS_ALG
