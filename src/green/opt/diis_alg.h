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

#include <complex>

// #include "com_diis_residual.h"
#include "diis_residual.h"
#include "iterative_optimizer.h"
#include "vector_space.h"

namespace mbpt::opt {

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
  template <typename Vector, typename Database>
  class diis_alg : public iterative_optimizer<Vector> {
  public:
    using iterative_optimizer<Vector>::_problem;
    using iterative_optimizer<Vector>::get_trust_norm;
    using iterative_optimizer<Vector>::put_trust_norm;

    // For now allow the external control on whether to do extrapolation
    bool _extrap;     // Perform extrapolation
    bool grow_xvsp;  // Controls simple growing of the xvec subspace.
                     // Needed if the residuals are r_i = x_i - x_{i-1}

    bool dm_incr = false;

  private:
    using MatrixXcd = Eigen::MatrixXcd;

    MatrixXcd            m_B;     // Overlap matrix of the residuals
    VectorXcd            m_C;     // Vector of extrapolation coefs
    std::complex<double> lambda;  // Lagrange multiplier for the constraint
    size_t               _max_subsp_size;
    // FIXME: need to initialize vspaces or pass them... Otherwise their strings are not defined
    VSpace<Vector, Database>*        _res_vsp;  // The subspace of residuals
    VSpace<Vector, Database>*        _x_vsp;    // The subspace of X vectors

    diis_residual<Vector, Database>* residual;  // Defines residual. Must be already initialized

    std::string                      diis_str = "DIIS: ";

  public:
    double get_err_norm() {
      size_t dim = m_B.cols();
      if (dim == 0) {
        std::cout << diis_str << "The error matrix has zero dimension" << std::endl;
        return 1.0;
      } else {
        return std::sqrt(std::abs(m_B(dim - 1, dim - 1)));
      }
    }

    void print_err_norm() {
      size_t dim = m_B.cols();
      if (dim == 0) {
        std::cout << diis_str << "The error matrix has zero dimension" << std::endl;
      } else {
        std::cout << diis_str << "The error norm = " << get_err_norm() << std::endl;
      }
    }

    void print_B() {
      std::cout << diis_str << "m_B:" << std::endl;
      std::cout << m_B << std::endl;
    }

    void print_C() {
      std::cout << diis_str << "Extrapolation coefs: " << std::endl;
      std::cout << m_C << std::endl;
    }

    // FIXME: probably is not the best function, since it is limited by "init"
    // functionality.
    void init(optimization_problem<Vector>* prob, size_t max_subsp_size_, bool extrap_, VSpace<Vector, Database>* x_vsp_,
              VSpace<Vector, Database>* res_vsp_, Vector& x_start, double trust) {
      _problem = prob;
      _x_vsp   = x_vsp_;
      _res_vsp = res_vsp_;
      residual->init(_problem, _x_vsp);
      // TODO: check initialization of the residual
      _max_subsp_size = max_subsp_size_;
      _extrap         = extrap_;
      _x_vsp->add_to_vspace(x_start);
      grow_xvsp        = true;
      this->_trust_norm = trust;
#if DIIS_DEBUG
      print_B();
#endif
    };

    void init(optimization_problem<Vector>* prob, diis_residual<Vector, Database>* residual_, size_t max_subsp_size_,
              bool extrap_, VSpace<Vector, Database>* x_vsp_, VSpace<Vector, Database>* res_vsp_, Vector& x_start, double trust) {
      if (!residual_->is_inited()) throw std::runtime_error("The residual is not initialized");

      _problem        = prob;
      residual       = residual_;
      _x_vsp          = x_vsp_;
      _res_vsp        = res_vsp_;
      _max_subsp_size = max_subsp_size_;
      _extrap         = extrap_;
      _x_vsp->add_to_vspace(x_start);
      grow_xvsp        = true;
      this->_trust_norm = trust;
#if DIIS_DEBUG
      print_B();
#endif
    };

    int next_step(Vector& vec) override {
      if (_x_vsp->size() == 0 || grow_xvsp) {
        std::cout << diis_str << "Growing subspace without extrapolation\n";
        _x_vsp->add_to_vspace(vec);
        _problem->put_x(vec);
        return 0;
      }
      // Normal execution
      if (_res_vsp->size() < _max_subsp_size) {
        _problem->put_x(vec);
        Vector res;
        if (!residual->get_diis_residual(res)) {
          std::cout << diis_str << "Could not get residual!!! ABORT!" << std::endl;
        }
        update_overlaps(res);  // the overlap with res is added in any case...

        _res_vsp->add_to_vspace(res);
        _x_vsp->add_to_vspace(vec);
      } else {  // The subspace is already of the maximum size
        std::cout << diis_str << "Reached maximum subspace. The first vector will be kicked out of the subspace." << std::endl;
        _res_vsp->purge_vec(0);  // can do it smarter and purge the one with the smallest coef
        _x_vsp->purge_vec(0);
        purge_overlap(0);

        _problem->put_x(vec);
        Vector res;
        if (!residual->get_diis_residual(res)) {
          std::cout << diis_str << "Could not get residual!!! ABORT!" << std::endl;
        }
        // TODO: treat linear deps
        update_overlaps(res);
        _res_vsp->add_to_vspace(res);
        _x_vsp->add_to_vspace(vec);
      }
      if (_extrap && (_res_vsp->size() > 1)) {
        compute_coefs(1);

        std::cout << diis_str << "Performing the DIIS extrapolation..." << std::endl;
        print_B();
        std::cout << diis_str << "Predicted extrapol (e,e) = " << m_C.dot(m_B * m_C) << std::endl;

        bool resc = this->limit_extrapol_coefs(m_C, diis_str);
        /* // TODO: Need to think whether we need this for now...
        if(!resc && !dm_incr) {
            std::cout << "MY: increasing the trust rad\n";
            double cur_trust = this->trust_norm;
            cur_trust *= 2.0;
            this->trust_norm = std::min(cur_trust, 1.0);
        }
        else {
            if(dm_incr) {
                std::cout << "MY: decreasing the trust rad\n";
                double cur_trust = this->trust_norm;
                cur_trust /= 2.0;
                this->trust_norm = std::min(cur_trust, 1.0);
            }
        }*/

        std::cout << diis_str << "Predicted extrapol (e,e) = " << m_C.dot(m_B * m_C) << std::endl;
        Vector result = _x_vsp->make_linear_comb(m_C);
        // std::cout << "Extrapolated vec overlap" << x_vsp->overlap(result,result) << std::endl;

        _problem->put_x(result);

      } else {
        // this->m_x = vec;
        _problem->put_x(vec);
      }
      return 0;
    }

  private:
    /**
     * Remove overlaps with the vector k
     *     The dimensions of the matrix m_B are shrinked by 1
     */
    void purge_overlap(const size_t k) {
      MatrixXcd Bnew(m_B.rows() - 1, m_B.cols() - 1);
      for (size_t i = 0, mi = 0; i < Bnew.rows(); i++, mi++) {
        if (i == k) ++mi;
        for (size_t j = 0, mj = 0; j < Bnew.cols(); j++, mj++) {
          if (j == k) ++mj;
          Bnew(i, j) = m_B(mi, mj);
        }
      }
      m_B = Bnew;
    }

    /**
     * Add overlaps with the incoming vector
     * The dimensions of the matrix m_B are extended by 1
     *
     * @param u
     */
    void update_overlaps(Vector& u) {
#if DIIS_DEBUG
      print_B();
      if (m_B.cols() > 1) {
        std::cout << "Before the update" << std::endl;
        print_B();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(m_B);
        std::cout << "evals: " << es.eigenvalues().transpose() << std::endl;
        std::cout << "updating..." << std::endl;
      }
#endif

      MatrixXcd Bnew = MatrixXcd::Zero(m_B.rows() + 1, m_B.cols() + 1);

      // Assing what is known already:
      for (size_t i = 0; i < m_B.rows(); i++) {
        for (size_t j = 0; j < m_B.cols(); j++) Bnew(i, j) = m_B(i, j);
      }

      // Evaluate new overlaps and add them to B:
      // Can ship this piece as a function with the vector space
      // for good parallelization
      for (size_t i = 0; i < m_B.cols(); i++) {
        Bnew(i, m_B.cols()) = _res_vsp->overlap(i, u);
        Bnew(m_B.cols(), i) = std::conj(Bnew(i, m_B.cols()));
      }
      Bnew(m_B.cols(), m_B.cols()) = _res_vsp->overlap(u, u);
      m_B                          = Bnew;
#if DIIS_DEBUG
      std::cout << "After the update" << std::endl;
      print_B();
#endif
    }

    // Simple, numerically unstable version
    void compute_coefs_simple() {
#if DIIS_DEBUG
      print_B();
#endif

      VectorXcd Cnew(m_B.cols());
      MatrixXcd B_cnstr(m_B.rows() + 1, m_B.cols() + 1);

      // Overlaps of error vectors
      for (size_t i = 0; i < m_B.rows(); i++) {
        for (size_t j = 0; j < m_B.cols(); j++) B_cnstr(i, j) = m_B(i, j);
      }

      // Constrants
      for (size_t i = 0; i < m_B.rows(); i++) {
        B_cnstr(i, m_B.cols()) = 1;
        B_cnstr(m_B.cols(), i) = 1;
      }

      B_cnstr(m_B.cols(), m_B.cols()) = 0;

      VectorXcd b                     = VectorXcd::Zero(B_cnstr.cols());
      b[B_cnstr.cols() - 1]           = 1;  // constraint

#if DIIS_DEBUG
      std::cout << "B_cnstr:" << std::endl;
      for (size_t i = 0; i < B_cnstr.rows(); i++) {
        for (size_t j = 0; j < B_cnstr.cols(); j++) std::cout << B_cnstr(i, j) << "  ";

        std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << "b" << std::endl;
      for (size_t i = 0; i < b.size(); i++) std::cout << b[i] << "  ";
      std::cout << std::endl;
#endif

      VectorXcd x = B_cnstr.colPivHouseholderQr().solve(b);
      for (size_t i = 0; i < m_B.rows(); i++) {
        Cnew[i] = x[i];
      }
      m_C    = Cnew;
      lambda = x[B_cnstr.cols() - 1];
      std::cout << "lambda = " << lambda << std::endl;
    }

    void compute_coefs(size_t option) {
      switch (option) {
        case 1:
          compute_coefs_c1();
          break;
        case 2:
          compute_coefs_c2();
          break;
        default:
          compute_coefs_c1();
      }
      print_cond();
      print_C();
    }

    // Compute and print the condition number
    void print_cond() {
      JacobiSVD<MatrixXcd> svd(m_B);
      double               cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
      std::cout << diis_str << "Condition number of the residual overlap matrix: " << cond << std::endl;
    }

    void compute_coefs_c1() {
#if DIIS_DEBUG
      print_B();
#endif

#pragma float_control(precise, on)  // Need accurate extrapolation coeffs
      MatrixXcd            B  = m_B.real();

      VectorXcd            bb = VectorXcd::Constant(B.cols(), 1.0);

      VectorXcd            x  = B.bdcSvd(ComputeFullU | ComputeFullV).solve(bb);
      std::complex<double> sum(0.0, 0.0);
      for (size_t i = 0; i < B.rows(); i++) {
        sum += x[i];
      }
      m_C = (x / sum).real();
    }

    void compute_coefs_c2() {
#if DIIS_DEBUG
      print_B();
#endif

#pragma float_control(precise, on)  // Need accurate extrapolation coeffs
      MatrixXcd           B  = m_B;

      VectorXcd           bb = VectorXcd::Constant(B.cols(), 1.0);

      BDCSVD<MatrixXcd>   svd(B, ComputeFullU | ComputeFullV);
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
      std::vector<double>::iterator it           = std::min_element(std::begin(errors), std::end(errors));
      size_t                        column_index = std::distance(std::begin(errors), it);
      m_C                                        = U.col(column_index);
    }
  };

}  // namespace opt

#endif  // GREEN_OPT_DIIS_ALG
