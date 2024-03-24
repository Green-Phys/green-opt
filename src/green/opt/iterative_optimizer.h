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

#ifndef GREEN_OPT_ITERATIVE_OPTIMIZER
#define GREEN_OPT_ITERATIVE_OPTIMIZER

#include "optimization_problem.h"

/** \brief Basic iterative optimizer
 *
 *  This class provides a basic structure of an iterative optimizer.
 *  Could be used for dispatching in future.
 *  **/

namespace green::opt {

  template <typename Vector, typename Derived>
  class iterative_optimizer {
  protected:
    double _trust_norm;  // limit of the coef vector norm

  public:
    virtual ~iterative_optimizer() = default;
    explicit iterative_optimizer(double trust_norm = 1.0) : _trust_norm(trust_norm) {}
    /**
     * Process vector for the next step of converegence acceleration.
     *
     * @tparam VS Type of the vector space
     * @tparam Res Type of residual evaluator
     * @param vec new vector to be processed
     * @param x_vsp values vector space
     * @param res_vsp residual vector space
     * @param residual functor to evaluate residual
     * @param problem current optimization problem
     */
    template <typename VS, typename Res>
    void next_step(Vector& vec, VS& x_vsp, VS& res_vsp, Res& residual, optimization_problem<Vector>& problem) {
      Derived::next_step(vec, x_vsp, res_vsp, residual, problem);
    }

    [[nodiscard]] double trust_norm() const { return _trust_norm; }
    double&              trust_norm() { return _trust_norm; }
  };
}  // namespace green::opt

#endif  // GREEN_OPT_ITERATIVE_OPTIMIZER
