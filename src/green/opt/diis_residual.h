/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_OPT_DIIS_RESIDUAL
#define GREEN_OPT_DIIS_RESIDUAL
#include "optimization_problem.h"
#include "vector_space.h"
/** \brief Implementation of the residuals for DIIS
 *
 *  The class provides a basic definition of the residual as a difference:
 *  \f[ r_i = x_i - x_{i-1} \f]
 *  This is an inheritable object---users can define their own residuals.
 *
 * **/

namespace mbpt::opt {

  template <typename Vector, typename Database, class Enable = void>
  class diis_residual {
  protected:
    optimization_problem<Vector>* _problem;
    VSpace<Vector, Database>*     _x_vsp;  // The subspace of X vectors
    bool                          _initialized = false;

  public:
         diis_residual() {}  // points must be initialized!

         diis_residual(optimization_problem<Vector>* prob) { _problem = prob; }

         diis_residual(optimization_problem<Vector>* prob, VSpace<Vector, Database>* x_space) { init(prob, x_space); }

    void init(optimization_problem<Vector>* prob, VSpace<Vector, Database>* x_space) {
      _problem        = prob;
      _x_vsp          = x_space;
      _initialized = true;
    }

    bool is_inited() const { return _initialized; }

    // Canonical implementation of the residual
    // as a difference between successive iterations.
    // Should be a reasonable default choice for residual definition.
    virtual bool get_diis_residual(Vector& res) {
      if (!_initialized) return false;
      if (_x_vsp->size() >= 2) {
        Vector last;
        _x_vsp->get_vec(_x_vsp->size() - 1, last);
        add(res, _problem->get_x(), last, std::complex<double>(-1.0, 0.0));  // vec - x_vsp.get_vec(vsp.size()-1);
        return true;
      } else {
        return false;
      }
    };
  };
}  // namespace opt

#endif  // GREEN_OPT_DIIS_RESIDUAL
