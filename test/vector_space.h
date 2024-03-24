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

#ifndef GREEN_OPT_VECTOR_SPACE
#define GREEN_OPT_VECTOR_SPACE

#include <Eigen/Dense>
#include <complex>

namespace green::opt {

  /**
   * Simple class for std::vector of complex values used for tests.
   */
  class VSpace {
  public:
    using Database                              = std::vector<std::vector<std::complex<double>>>;
    using Vector                                = std::vector<std::complex<double>>;
                                       VSpace() = default;

                                       VSpace(Database& db) : m_dbase(db) {}

    Vector&                            get(size_t i) { return m_dbase[i]; }

    void                               get(size_t i, Vector& Vec) { Vec = m_dbase[i]; };

    void                               add(Vector& a) { m_dbase.push_back(a); }

    std::complex<double>               overlap(size_t i, size_t j) { return overlap(m_dbase[i], m_dbase[j]); }

    std::complex<double>               overlap(size_t i, const Vector& a) { return overlap(m_dbase[i], a); }

    [[nodiscard]] std::complex<double> overlap(const Vector& a, const Vector& b) const {
      std::complex<double> s(0.0, 0.0);
      for (size_t i = 0; i < a.size(); i++) {
        s += std::conj(a[i]) * b[i];
      }
      return s;
    }

    [[nodiscard]] size_t size() const { return m_dbase.size(); };

    void                 purge(size_t k) { m_dbase.erase(m_dbase.begin() + k); }

    Vector               make_linear_comb(const Eigen::VectorXcd& C) {
      if (m_dbase.size() == 0) {
        return {0, 0.0};
      }
      Vector r(m_dbase[0].size(), 0);
      for (size_t i = 0; i < m_dbase.size() && i < C.size(); i++) {
        Vector               vec_i = get(size() - 1 - i);
        std::complex<double> coeff = C(C.size() - 1 - i);
        std::transform(vec_i.begin(), vec_i.end(), vec_i.begin(), [coeff](const std::complex<double>& x) { return coeff * x; });
        std::transform(vec_i.begin(), vec_i.end(), r.begin(), r.begin(),
                                     [](const std::complex<double>& x, const std::complex<double>& y) { return x + y; });
      }
      return r;
    }

  private:
    Database m_dbase;
  };

  inline void add(std::vector<std::complex<double>>& res, std::vector<std::complex<double>>& a,
                  std::vector<std::complex<double>>& b, std::complex<double> c) {
    assert(a.size() == b.size());
    res.resize(a.size());
    std::transform(a.begin(), a.end(), b.begin(), res.begin(),
                   [c](const std::complex<double>& A, const std::complex<double>& B) { return A + B * c; });
  }

  inline std::vector<std::complex<double>> add(std::vector<std::complex<double>>& a, std::vector<std::complex<double>>& b,
                                               std::complex<double> c) {
    assert(a.size() == b.size());
    std::vector<std::complex<double>> res(a.size(), 0.0);
    add(res, a, b, c);
    return res;
  }

}  // namespace green::opt
#endif  // GREEN_OPT_VECTOR_SPACE
