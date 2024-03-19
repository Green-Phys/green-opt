/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_OPT_VECTOR_SPACE
#define GREEN_OPT_VECTOR_SPACE

#include <Eigen/Dense>
#include <complex>

namespace mbpt::opt {

  using namespace Eigen;

  template <typename Vector, typename Database, class Enable = void>
  class VSpace {
  private:
    size_t   m_size;
    Database m_dbase;

  public:
                         VSpace() { m_size = 0; }

                         VSpace(Database& db) : m_dbase(db) { m_size = 0; }

    Vector&              get_vec(const size_t i);

    void                 get_vec(const size_t i, Vector& Vec);

    void                 add_to_vspace(Vector& a);

    std::complex<double> overlap(const size_t i, const size_t j);

    std::complex<double> overlap(const size_t i, const Vector& a);

    std::complex<double> overlap(const Vector& a, const Vector& b);

    size_t               size() { return m_size; };

    void                 purge_vec(const size_t k);

    virtual Vector       make_linear_comb(const VectorXcd& C) {
      Vector r;
      r.zero();
      for (size_t i = 0; i < m_size; i++) r.add(get_vec(i), C[i]);
      return r;
    }
  };

  template <typename Vector>
  Vector add(Vector& a, Vector& b, std::complex<double> c);

  template <typename Vector>
  void add(Vector& res, Vector& a, Vector& b, std::complex<double> c);

#if 0

typedef std::vector<std::complex<double> > cvec;

template<>
class VSpace<cvec, std::vector<cvec> > {
private:
    size_t m_size;
    std::vector<cvec> m_dbase;

public:

    VSpace() {
        m_size = 0;
    }

    VSpace(std::vector<cvec >& db) : m_dbase(db) {
        m_size = 0;
    }

    cvec& get_vec(const size_t i) {return m_dbase[i]; }

    void add_to_vspace(cvec& a) {
        m_dbase.push_back(a); m_size++; 
    };

    std::complex<double> overlap(const size_t i, const size_t j) {
        return overlap(get_vec(i), get_vec(j));
    };

    std::complex<double> overlap(const size_t i, const cvec& a) {
        return overlap(get_vec(i), a);
    };

    std::complex<double> overlap(const cvec& a, const cvec& b) {
        std::complex<double> s(0.0,0.0);
        for(size_t i = 0; i < a.size(); i++) {
            s += std::conj(a[i]) * b[i];
        }
    }

    size_t size() {
        return m_size; 
    };

    void purge_vec(const size_t k) {
        m_dbase.erase(m_dbase.begin()+k);
        m_size--;
    };

    cvec make_linear_comb(VectorXcd& C) {
         cvec r;
         for(size_t j = 0; j < r.size(); j++) r[j] = 0;

         for(size_t i = 0; i < m_size; i++)   { // loop over basis
             cvec vec_i = get_vec(i);
             for(size_t j = 0; j < r.size(); j++) // loop over components of vecs
                 r[j] += (vec_i)[j]*C[i];
         }
         return r;
     }

};
#endif

}  // namespace opt
#endif  // GREEN_OPT_VECTOR_SPACE
