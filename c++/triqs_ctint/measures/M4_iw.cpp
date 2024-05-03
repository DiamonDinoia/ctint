#include "./M4_iw.hpp"
#include <cmath>

#include <immintrin.h> // for AVX2 intrinsics

namespace triqs_ctint::measures {

  M4_iw::M4_iw(params_t const &params_, qmc_config_t const &qmc_config_, container_set *results)
     : params(params_), qmc_config(qmc_config_), buf_arrarr(params_.n_blocks()) {

    // Construct Matsubara mesh
    mesh::imfreq iw_mesh{params.beta, Fermion, params.n_iw_M4};
    mesh::prod<imfreq, imfreq, imfreq> M4_iw_mesh{iw_mesh, iw_mesh, iw_mesh};

    // Init measurement container and capture view
    results->M4_iw = make_block2_gf(M4_iw_mesh, params.gf_struct);
    M4_iw_.rebind(results->M4_iw.value());
    M4_iw_() = 0;

    // Construct Matsubara mesh for temporary Matrix
    mesh::imfreq iw_mesh_large{params.beta, Fermion, 3 * params.n_iw_M4};
    mesh::prod<imfreq, imfreq> M_mesh{iw_mesh_large, iw_mesh};

    // Initialize intermediate scattering matrix
    M = block_gf{M_mesh, params.gf_struct};

    // Create nfft buffers
    for (int bl : range(params.n_blocks())) {
      auto init_target_func = [&](int i, int j) {
        return nfft_buf_t<2>{slice_target_to_scalar(M[bl], i, j).data(), params.nfft_buf_size, params.beta};
      };
      buf_arrarr(bl) = array_adapter{M[bl].target_shape(), init_target_func};
    }
  }

  void M4_iw::accumulate(mc_weight_t sign) {
    // Accumulate sign
    Z += sign;

    // Calculate intermediate scattering matrix
    M() = 0;
    for (int bl : range(params.n_blocks()))
      //for (auto &[c_i, cdag_j, Ginv1] : qmc_config.dets[b1]) // FIXME c++17
      foreach (qmc_config.dets[bl],
               [&](c_t const &c_i, cdag_t const &cdag_j, auto const &Ginv_ji) { // Care for negative frequency in c transform (for M-objects)
                 buf_arrarr(bl)(cdag_j.u, c_i.u).push_back({double(cdag_j.tau), params.beta - double(c_i.tau)}, -Ginv_ji);
               })
        ;
    for (auto &buf_arr : buf_arrarr)
      for (auto &buf : buf_arr) buf.flush(); // Flush remaining points from all buffers

    auto const &iw_mesh = std::get<0>(M4_iw_(0, 0).mesh());
    const auto v_sign = _mm512_set1_pd(sign);
    for (int bl1 : range(params.n_blocks())) // FIXME c++17 Loops
      for (int bl2 : range(params.n_blocks())) {
        int bl1_size   = M[bl1].target_shape()[0];
        int bl2_size   = M[bl2].target_shape()[0];
        auto const &M1 = M[bl1];
        auto const &M2 = M[bl2];
        auto &M4       = M4_iw_(bl1, bl2);
        for (auto iw1 : iw_mesh)
          for (auto iw2 : iw_mesh)
#pragma clang l
            for (auto iw3 : iw_mesh) {
              const auto total_size = bl1_size * bl1_size * bl2_size * bl2_size;
              const auto iw4        = iw1 + iw3 - iw2;
              const auto remainder = total_size % 4;
              const auto inv_bl1_size_cubed = 1 / (bl1_size * bl2_size * bl2_size);
              const auto inv_bl2_size_squared = 1 / (bl2_size * bl2_size);
              const auto inv_bl1_size = 1 / bl1_size;
              const auto inv_bl1_size_squared = 1 / (bl1_size * bl1_size);
              const auto bl12 = (bl1_size * bl2_size);
              for (auto index = 0; index < total_size - remainder; index += 4) {
                // Load complex numbers as pairs of doubles
                const auto li = index * inv_bl1_size_cubed;
                const auto jk = (index * inv_bl2_size_squared) % bl12;
                const auto lk = (index * inv_bl1_size) % bl12;
                const auto ji = index * inv_bl1_size_squared;
                const auto M1val_v = _mm512_loadu_pd(reinterpret_cast<double*>(M1[iw2.value(), iw1].data()+ji));
                const auto M2_lk_v = _mm512_loadu_pd(reinterpret_cast<double*>(M2[iw4, iw3].data()+lk));
                const auto M1_li_v = _mm512_loadu_pd(reinterpret_cast<double*>(M1[iw4, iw1].data()+li));
                const auto M2_jk_v = _mm512_loadu_pd(reinterpret_cast<double*>(M2[iw2.value(), iw3].data()+jk));
                const auto M4_v = _mm512_loadu_pd(reinterpret_cast<double*>(M4[iw1, iw2, iw3].data()+index));
                // Separate real and imaginary parts
                const auto idx_even = _mm512_set_epi64(6, 4, 2, 0, 6, 4, 2, 0);
                const auto idx_odd = _mm512_set_epi64(7, 5, 3, 1, 7, 5, 3, 1);
                const auto M1val_real_v = _mm512_permutexvar_pd(idx_even, M1val_v);
                const auto M1val_imag_v = _mm512_permutexvar_pd(idx_odd, M1val_v);
                const auto M2_lk_real_v = _mm512_permutexvar_pd(idx_even, M2_lk_v);
                const auto M2_lk_imag_v = _mm512_permutexvar_pd(idx_odd, M2_lk_v);
                const auto M1_li_real_v = _mm512_permutexvar_pd(idx_even, M1_li_v);
                const auto M1_li_imag_v = _mm512_permutexvar_pd(idx_odd, M1_li_v);
                const auto M2_jk_real_v = _mm512_permutexvar_pd(idx_even, M2_jk_v);
                const auto M2_jk_imag_v = _mm512_permutexvar_pd(idx_odd, M2_jk_v);
                const auto M4_real_v = _mm512_permutexvar_pd(idx_even, M4_v);
                const auto M4_imag_v = _mm512_permutexvar_pd(idx_odd, M4_v);
                // Perform the operations
                auto result_real_v = _mm512_add_pd(M4_real_v, _mm512_mul_pd(v_sign, _mm512_sub_pd(_mm512_mul_pd(M1val_real_v, M2_lk_real_v), _mm512_mul_pd(M1val_imag_v, M2_lk_imag_v))));
                auto result_imag_v = _mm512_add_pd(M4_imag_v, _mm512_mul_pd(v_sign, _mm512_add_pd(_mm512_mul_pd(M1val_real_v, M2_lk_imag_v), _mm512_mul_pd(M1val_imag_v, M2_lk_real_v))));
                result_real_v = __builtin_expect(!!bl1 == bl2, 0) ? _mm512_sub_pd(result_real_v, _mm512_mul_pd(v_sign, _mm512_sub_pd(_mm512_mul_pd(M1_li_real_v, M2_jk_real_v), _mm512_mul_pd(M1_li_imag_v, M2_jk_imag_v)))) : result_real_v;
                result_imag_v = __builtin_expect(!!bl1 == bl2, 0) ? _mm512_sub_pd(result_imag_v, _mm512_mul_pd(v_sign, _mm512_add_pd(_mm512_mul_pd(M1_li_real_v, M2_jk_imag_v), _mm512_mul_pd(M1_li_imag_v, M2_jk_real_v)))) : result_imag_v;
                // Pack real and imaginary parts back together
                const auto result_v = _mm512_unpacklo_pd(result_real_v, result_imag_v);
                // Store the result back into M4
                _mm512_storeu_pd(reinterpret_cast<double*>(M4[iw1, iw2, iw3].data()+index), result_v);
              }
              for (int index = total_size - remainder; index < total_size; ++index) {
                const auto li = index / (bl1_size * bl2_size * bl2_size);
                const auto jk = (index / (bl2_size * bl2_size)) % (bl1_size * bl2_size);
                const auto lk = (index / bl1_size) % (bl2_size * bl2_size);
                const auto ji = index % (bl1_size * bl1_size);
                auto M1val    = M1[iw2.value(), iw1].data()[ji];
                M4[iw1, iw2, iw3].data()[index] += sign * M1val * M2[iw4, iw3].data()[lk];
                M4[iw1, iw2, iw3].data()[index] -= bl1 == bl2 ? sign * M1[iw4, iw1].data()[li] * M2[iw2.value(), iw3].data()[jk] : 0 ;
              }
            }
      }
  }

  void M4_iw::collect_results(mpi::communicator const &comm) {
    // Collect results and normalize
    Z      = mpi::all_reduce(Z, comm);
    M4_iw_ = mpi::all_reduce(M4_iw_, comm);
    M4_iw_ = M4_iw_ / (Z * params.beta);
  }

} // namespace triqs_ctint::measures
