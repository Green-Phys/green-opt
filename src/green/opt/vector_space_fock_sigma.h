/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */


#ifndef GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA
#define GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA

#include <Eigen/Dense>
#include <complex>
#include "vector_space.h"

#include <stdexcept>

#include <type_traits>

// #include "transformer_t.h"

namespace mbpt::opt {


using namespace Eigen;

/** \brief FockSigma
 *  
 *  The class provides a vector definition for combined Fock matrix and Self-energy.
 *
 * **/

class FockSigma {
private:
    ztensor<4> m_Fock;
    ztensor<5> m_Sigma;
public:
    FockSigma() {}
    FockSigma(const FockSigma & rhs) : m_Fock(rhs.m_Fock.copy()), m_Sigma(rhs.m_Sigma.copy()) {}

    FockSigma(const ztensor<4>& Fock_, const ztensor<5>& Sigma_) : m_Fock(Fock_.copy()), m_Sigma(Sigma_.copy()) {}

    FockSigma& operator =(const FockSigma& rhs) {
      m_Fock = rhs.m_Fock.copy();
      m_Sigma = rhs.m_Sigma.copy();
      return *this;
    }

    ztensor<4>& get_fock() {
        return m_Fock;
    }
    ztensor<5>& get_sigma() {
        return m_Sigma;
    }
    void set_fock(ztensor<4>& F_) {
        m_Fock = F_.copy();
    }
    void set_sigma(ztensor<5>& S_) {
        m_Sigma = S_.copy();
    }
    void set_fock_sigma(ztensor<4>& F_, ztensor<5>& S_) {
        set_fock(F_);
        set_sigma(S_);
    }
    const ztensor<4>& get_fock() const {
        return m_Fock;
    }
    const ztensor<5>& get_sigma() const {
        return m_Sigma;
    }

    std::complex<double>* get_fock_data()  {
        return m_Fock.data();
    }
    std::complex<double>* get_sigma_data()  {
        return m_Sigma.data();
    }
    const std::array < size_t, 4>& get_fock_shape() const  {
        return m_Fock.shape();
    }
    const std::array < size_t, 5>& get_sigma_shape() const  {
        return m_Sigma.shape();
    }

    void set_zero() {
        m_Fock.set_zero();
        m_Sigma.set_zero();
    }

    FockSigma operator*=(std::complex<double> c)  {
        m_Fock *= c;
        m_Sigma *= c;
        return *this;
    }

    FockSigma operator+=(FockSigma & vec)  {
        m_Fock += vec.get_fock();
        m_Sigma += vec.get_sigma();
        return *this;
    }

    FockSigma operator+=(FockSigma && vec)  {
        m_Fock += vec.get_fock();
        m_Sigma += vec.get_sigma();
        return *this;
    }
    
};


/** 
 * Evaluation of the commutator in the tau space between G and G_0^{-1} - Sigma
 *
 * **/
    void commutator_t(const transformer_t& ft, ztensor<5>& C_t, ztensor<5>& G_t_,
                      FockSigma& FS_t, double mu, ztensor<4>& S) {
        size_t nts = G_t_.shape()[0];
        size_t ns = G_t_.shape()[1];
        size_t nk = G_t_.shape()[2];
        size_t nao = G_t_.shape()[4];
        size_t nw = ft.wsample_fermi().size();

        ztensor<2> I(nao, nao);
        ztensor<3> Sigma_w(nw, nao, nao);
        ztensor<3> Sigma_k(nts, nao, nao);
        ztensor<3> G_w(nw, nao, nao);
        ztensor<3> C_w(nw, nao, nao);
        ztensor<3> C_t_slice(nts, nao, nao);
        ztensor<3> G_t(nts, nao, nao);
        // PP: This one is needed if C_t is not allocated
        // (which I assume is the case since all the params, 
        //  such as nts, ns, etc should not be known in the abstract classes)
        ztensor<5> C_t_full(nts, ns, nk, nao, nao);

        // k-points and spin are moved as an outer loop, 
        // because in future it is possible to make it MPI-parallel
        for(size_t isk = 0; isk < ns*nk; isk++) {
            size_t is = isk / nk;
            size_t ik = isk % nk;
            Sigma_k.set_zero();
            for (size_t it = 0; it < nts; ++it) matrix(Sigma_k(it)) = matrix(FS_t.get_sigma()(it, is, ik));
            ft.tau_to_omega(Sigma_k, Sigma_w, 1);
            for (size_t it = 0; it < nts; ++it) matrix(G_t(it)) = matrix(G_t_(it, is, ik));
            ft.tau_to_omega(G_t, G_w, 1);
            for(size_t iw = 0; iw < nw; iw++) {
                // Take nao x nao matrices at certain omega, spin, and k-point
                std::complex<double> muomega = ft.omega(ft.wsample_fermi()[iw], 1) + mu;
                MMatrixXcd MO(S.data()+isk*nao*nao, nao, nao);
                MMatrixXcd MI(I.data(), nao, nao);
                MMatrixXcd MC(C_w.data()+iw*nao*nao, nao, nao);
                MMatrixXcd MF(FS_t.get_fock_data()+isk*nao*nao, nao, nao);
                MMatrixXcd MS(Sigma_w.data()+iw*nao*nao, nao, nao);
                MMatrixXcd MG(G_w.data()+iw*nao*nao, nao, nao);
                MI = muomega*MO - MF - MS;
                MC = MG*MI - MI*MG;
            }
            ft.omega_to_tau(C_w, C_t_slice, 1);
            for (size_t it = 0; it < nts; ++it) matrix(C_t_full(it, is, ik)) = matrix(C_t_slice(it));
        }
        C_t = C_t_full; // copy forcing re-allocation of C_t
    }

template<typename T> 
void add(T& res, T& b, std::complex<double> c) {
    res = b;
    res *= c;
}

template<typename T> 
void add(T& res, T& a, T& b, std::complex<double> c) {
    res = b;
    res *= c;
    res += a;
}

/** \brief Vector space implementation
 *  
 *  This template specialization considers cases when the Vector type is 
 *  ztensor<4> (Fock matrix), ztensor<5> (Self-energy), or FockSigma (both). 
 *
 *  The class stores the subspace on disk in HDF5 format. 
 *  Publickly available functionality: access to vectors, 
 *  addition to the vector space, removal from the vector space, 
 *  evaluation of Euclidian overlaps (without any affine metrics/preconditioners), 
 *  evaluation of a linear combination of vectors.
 *
 *  TODO: MPI parallelization; "move" operation can be done MUCH more efficiently with HDF5.
 * **/

template<typename Vector>
class VSpace<Vector, std::string, typename std::enable_if< 
                 std::is_same<Vector, ztensor<4> >::value ||
                 std::is_same<Vector, ztensor<5> >::value ||
                 std::is_same<Vector, FockSigma >::value 
                >::type  > {
private:
    size_t m_size;
    std::string m_dbase; // Name of the file where the vectors will be saved
    std::string vecname; // Name of the vector to be saved

    constexpr static bool is_ztensor = (std::is_same<Vector, ztensor<4> >::value) || 
                                (std::is_same<Vector, ztensor<5> >::value); 

/** \brief Read vector \f[i\f] from file
 *  \param i
 * **/
template <typename T = Vector, typename std::enable_if< 
    std::is_same< T, Vector >::value &&
    (std::is_same< Vector, ztensor<4> >::value || 
     std::is_same< Vector, ztensor<5> >::value), bool >::type = true>
    void read_from_dbase(const size_t i, T& Vec) {
        green::h5pp::archive vsp_ar(m_dbase, "r");
        vsp_ar["vec" + std::to_string(i) + "/" + vecname + "/data"] >> Vec;
        vsp_ar.close();
    }

template <typename T = Vector, typename std::enable_if< 
    std::is_same< T, Vector >::value &&
    std::is_same< Vector, FockSigma >::value, bool >::type = true >
    void read_from_dbase(const size_t i, T& Vec) {
        green::h5pp::archive vsp_ar(m_dbase, "r");
        vsp_ar["vec" + std::to_string(i) + "/" + "Fock" + "/data"] >> Vec.get_fock();
        vsp_ar["vec" + std::to_string(i) + "/" + "Selfenergy" + "/data"] >> Vec.get_sigma();
        vsp_ar.close();
    }


/** \brief Write vector to position \f[i\f] in the file
 *  \param i
 * **/
    template < typename T = Vector, typename std::enable_if< 
    std::is_same< T, Vector >::value &&
    (std::is_same< Vector, ztensor<4> >::value || 
     std::is_same< Vector, ztensor<5> >::value), bool >::type = true >
    void write_to_dbase(const size_t i, Vector& Vec) {
        green::h5pp::archive vsp_ar(m_dbase, "a");
        vsp_ar["vec" + std::to_string(i) + "/" + vecname + "/data"] << Vec;
        vsp_ar.close();
    }

    template < typename T = Vector, typename std::enable_if< 
    std::is_same< T, Vector >::value &&
    std::is_same< Vector, FockSigma >::value, bool >::type = true >
    void write_to_dbase(const size_t i, Vector& Vec) {
        green::h5pp::archive vsp_ar(m_dbase, "a");
        vsp_ar["vec" + std::to_string(i) + "/" + "Fock" + "/data"] << Vec.get_fock();
        vsp_ar["vec" + std::to_string(i) + "/" + "Selfenergy" + "/data"] << Vec.get_sigma();
        vsp_ar.close();
    }

public:

    VSpace() {
        m_size = 0;
        vecname = (std::is_same<Vector, ztensor<4> >::value) ? "Fock" : 
                  (std::is_same<Vector, ztensor<5> >::value) ? "Selfenergy" : 
                  "FockSelfenergy";
    }

    VSpace(const std::string& db) : m_dbase(db)  {
        m_size = 0;
        //vecname = (std::is_same<Vector, ztensor<4> >::value) ? "Fock" : "Selfenergy";
        vecname = (std::is_same<Vector, ztensor<4> >::value) ? "Fock" : 
                  (std::is_same<Vector, ztensor<5> >::value) ? "Selfenergy" : 
                  "FockSelfenergy";
    }

    Vector get_vec(const size_t i) {
        Vector Vec;
        get_vec(i, Vec);

        return Vec;
    }

    void get_vec(const size_t i, Vector& Vec) {
        if(i >= m_size) {
            throw std::runtime_error("Vector index of the VSpace container is out of bounds");
        }
        read_from_dbase(i, Vec);
    }

    void add_to_vspace(Vector& Vec) {
        write_to_dbase(m_size, Vec);
        m_size++;
    }

    std::complex<double> overlap(const size_t i, const Vector& Vec_u) {
        Vector Vec_i = get_vec(i);
        return overlap(Vec_i, Vec_u);
    }

    std::complex<double> overlap(const size_t i, const size_t j) {
        Vector Vec_i = get_vec(i);
        Vector Vec_j = get_vec(j);
        return overlap(Vec_i, Vec_j);
    }

    template < typename T = Vector, typename std::enable_if<
        std::is_same< T, Vector >::value &&
        (std::is_same< Vector, ztensor<4> >::value || 
         std::is_same< Vector, ztensor<5> >::value), bool >::type = true >
    std::complex<double> overlap(const T& Vec_v, const T& Vec_u) {
            CMcolumn MVec_v(Vec_v.data(), Vec_v.size()); 
            CMcolumn MVec_u(Vec_u.data(), Vec_u.size());
            return MVec_v.dot(MVec_u);
    }
#if 1
    template < typename T = Vector, typename std::enable_if<
        std::is_same< T, Vector >::value &&
        std::is_same< Vector, FockSigma >::value, bool  >::type = true >
    std::complex<double> overlap(const T& Vec_v, const T& Vec_u) {
            const ztensor<4>& Fock_ref_v = Vec_v.get_fock();
            const ztensor<4>& Fock_ref_u = Vec_u.get_fock();
            CMcolumn MFVec_v(Fock_ref_v.data(), Fock_ref_v.size()); 
            CMcolumn MFVec_u(Fock_ref_u.data(), Fock_ref_u.size()); 
            const ztensor<5>& Sigma_ref_v = Vec_v.get_sigma();
            const ztensor<5>& Sigma_ref_u = Vec_u.get_sigma();
            CMcolumn MSVec_v(Sigma_ref_v.data(), Sigma_ref_v.size()); 
            CMcolumn MSVec_u(Sigma_ref_u.data(), Sigma_ref_u.size()); 

            // TODO: think whether a rescaling of Sigma is needed...
            return MFVec_v.dot(MFVec_u) + MSVec_v.dot(MSVec_u);
    }
#endif

    size_t size() {
        return m_size; 
    };

#if 1
// TODO implement "move" operation in the ALPS
    void purge_vec(const size_t i) {
        if(i >= m_size) {
            throw std::runtime_error("Vector index of the VSpace container is out of bounds");
        }
        if(m_size == 0) {
            throw std::runtime_error("VSpace container is of zero size, no vectors can be deleted");
        }
        Vector Vec;
        for(size_t j = i+1; j < size(); j++) {
            read_from_dbase(j, Vec);
            write_to_dbase(j-1, Vec);
        }
        //green::h5pp::archive vsp_ar_w(m_dbase, "w");
        //vsp_ar_w.delete_group("vec" + std::to_string(size()-1));
        
        m_size--; 
        
    }
#endif
#if 1
    virtual Vector make_linear_comb(const VectorXcd& C) {
         Vector r;
         get_vec(size()-1, r); // this is needed to initialize r
         r.set_zero();
         for(size_t i = 0; i < m_size && i < C.size(); i++) {
             Vector vec_i;
             get_vec(size()-1-i, vec_i);
             std::complex<double> coeff = C(C.size()-1-i);
             vec_i *= coeff;
             r += vec_i;
         }
         return r;
     }
#endif

};




} // namespace
#endif //  GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA
