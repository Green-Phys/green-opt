/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_OPT_DIIS_RESIDUAL
#define GREEN_OPT_DIIS_RESIDUAL
#include "optimization_problem.h"
/** \brief Implementation of the residuals for DIIS
 *
 *  The class provides a basic definition of the residual as a difference:
 *  \f[ r_i = x_i - x_{i-1} \f]
 *  This is an inheritable object---users can define their own residuals.
 *
 * **/

namespace green::opt {

  template <typename Vector>
  class diis_residual {
  public:
    diis_residual() {}  // points must be initialized!

    // Canonical implementation of the residual
    // as a difference between successive iterations.
    // Should be a reasonable default choice for residual definition.
    template<typename VS>
    bool operator()(Vector& res, VS& x_vsp, optimization_problem<Vector>& problem) {
      if (x_vsp.size() >= 2) {
        Vector last;
        x_vsp.get(x_vsp.size() - 1, last);
        add(res, problem.x(), last, std::complex<double>(-1.0, 0.0));  // vec - x_vsp.get_vec(vsp.size()-1);
        return true;
      }
      return false;
    };
  };
}  // namespace mbpt::opt

#endif  // GREEN_OPT_DIIS_RESIDUAL
