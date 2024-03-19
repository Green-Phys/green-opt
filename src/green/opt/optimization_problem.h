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

#ifndef GREEN_OPT_OPTIMIZATION_PROBLEM
#define GREEN_OPT_OPTIMIZATION_PROBLEM

namespace mbpt::opt {

  /** \brief Basic optimization problem
   *
   *  This class provides a basic structure of an optimization problem.
   *  Allows one to abstract the iterative solver from a particular problem.
   *  The real optimization problem can inherit from this class.
   *
   **/
  template <typename Vector>
  class optimization_problem {
  protected:
    size_t iter;
    Vector m_x;

  public:
    optimization_problem() {}

    Vector  get_x() && { return m_x; }

    Vector& get_x() & { return m_x; }  // l-value version

    void    get_x(Vector& vec) { vec = m_x; }

    void    put_x(const Vector& vec) { m_x = vec; }
  };
}  // namespace opt

#endif  // GREEN_OPT_OPTIMIZATION_PROBLEM
