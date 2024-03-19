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

#ifndef COM_DIIS_RESIDUAL
#define COM_DIIS_RESIDUAL
#include "optimization_problem.h"
#include "vector_space.h"
// #include "vector_space_fock_sigma.h"

namespace mbpt::opt {
  // So far it is limited to FockSigma cases only
  /** \brief Implementation of the residuals for DIIS
   *
   *  The class provides a basic definition of the residual as a commutator:
   *  \f[ r_i = [G_i, G_0^{-1} - \Sigma_i] \f]
   *
   * **/
  template <typename Database>
  class com_diis_residual : public diis_residual<FockSigma, Database> {
  protected:
    using Vector = FockSigma;
    using diis_residual<Vector, Database>::problem;
    using diis_residual<Vector, Database>::x_vsp;  // The subspace of X vectors
    using diis_residual<Vector, Database>::is_initialized;

    transformer_t& m_ft;
    ztensor<4>&    m_S;  // Overlap matrix

    // VSpace<ztensor<5>, Database>* G_vsp;   // The subspace of Green's functions in tau
    double     mu;          // Chemical potential
    bool       double_com;  // Disabled for now

    ztensor<5> G_incoming;  // Need to store somewhere the incoming Green's function before
                            // the tensor in sc_loop.cpp gets destroyed.

  public:
    void upload_g(ztensor<5>& G_) { G_incoming = G_; }

         com_diis_residual(optimization_problem<Vector>* prob, VSpace<Vector, Database>* x_space,
                           // VSpace<ztensor<5>, Database>* g_space,
                           transformer_t& ft, ztensor<4>& S, double mu_, bool double_com_) :
        m_ft(ft), m_S(S), mu(mu_), double_com(double_com_) {
      init(prob, x_space);
    }

    virtual void init(optimization_problem<Vector>* prob, VSpace<Vector, Database>* x_space) {
      // VSpace<ztensor<5>, Database>* g_space) {
      problem = prob;
      x_vsp   = x_space;
      // G_vsp = g_space;
      is_initialized = true;
    }

    void update_mu(double mu_) { mu = mu_; }

    // Commutator residual
    // This may not be the most memory-efficient implementation...
    virtual bool get_diis_residual(Vector& res) {
      if (!is_initialized) {
        std::cout << "Commutator residual is not initialized!!!" << std::endl;
        return false;
      }
      if (x_vsp->size() >= 2) {
        // Warning! Sigma here is in tau!
        Vector x_last = problem->get_x();
        // x_vsp->get_vec(x_vsp->size()-1, x_last);
        //  Warning! G here is in tau!
        // ztensor<5> G_t;
        // G_vsp->get_vec(G_vsp->size()-1, G_t);

        ztensor<5> C_t;
        commutator_t(m_ft, C_t, G_incoming, x_last, mu, m_S);

        ztensor<4> Fz = x_last.get_fock();
        Fz.set_zero();
        res.set_fock_sigma(Fz, C_t);

        return true;
      } else {
        return false;
      }
    };
  };
}  // namespace opt

#endif  // COM_DIIS_RESIDUAL
