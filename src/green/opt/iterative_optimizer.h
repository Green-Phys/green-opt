/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_OPT_ITERATIVE_OPTIMIZER
#define GREEN_OPT_ITERATIVE_OPTIMIZER

#include <iostream>
#include "optimization_problem.h"

/** \brief Basic iterative optimizer
 *
 *  This class provides a basic structure of an iterative optimizer.
 *  Could be used for dispatching in future.
 *  **/

namespace mbpt::opt {

  template <typename Vector>
  class iterative_optimizer {
  protected:
    size_t                        _iter;
    optimization_problem<Vector>* _problem;

    double                        _trust_norm = 1;  // limit of the coef vector norm

  public:
                iterative_optimizer() : _iter(0) {}
    virtual int next_step(Vector& vec) { return 0; };
    // virtual double conv_criterion() = 0;
    Vector  get_x() && { return _problem->get_x(); }  // C++11 compiler makes it with r-value refs
    Vector& get_x() & { return _problem->get_x(); }

    double          get_trust_norm() const { return _trust_norm; }

    void            put_trust_norm(double trust_norm_) { _trust_norm = trust_norm_; }

    virtual bool    limit_extrapol_coefs(VectorXcd& m_C, const std::string& method_str) {
      // std::cout << "Trust norms: " << trust_norm  << "  "  << std::endl;
      bool                 rescaling = false;
      VectorXcd            C_delta   = m_C;
      std::complex<double> tmp       = C_delta[C_delta.size() - 1];
      C_delta[C_delta.size() - 1]    = -(1.0 - tmp);
      if (C_delta.norm() > _trust_norm) {
        rescaling   = true;
        double norm = C_delta.norm();
        std::cout << method_str << "The coefficient norm is too large: " << norm << std::endl;
        std::cout << method_str << "Rescaling... " << std::endl;
        C_delta *= (_trust_norm / norm);
        m_C = C_delta;
        m_C[m_C.size() - 1] += 1.0;
      }
      // FockSigma delta = x_vsp->make_linear_comb(C_delta);
      // FockSigma result = x_vsp->make_linear_comb(m_C);
      // problem->put_x(result);
      return rescaling;
    }
  };
}  // namespace opt

#endif  // GREEN_OPT_ITERATIVE_OPTIMIZER
