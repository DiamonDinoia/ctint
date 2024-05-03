#include "./M4_iw.hpp"
#include <cmath>

#include <immintrin.h> // for AVX2 intrinsics

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> separate_real_imaginary(const __m512d v1, const __m512d v2) noexcept {
  // Create index vectors for separating real and imaginary parts
  static const __m512i idx_real = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  static const __m512i idx_imag = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  // Separate real and imaginary parts
  __m512d real_v = _mm512_permutex2var_pd(v1, idx_real, v2);
  __m512d imag_v = _mm512_permutex2var_pd(v1, idx_imag, v2);
  return {real_v, imag_v};
}
__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> interleave_vectors(__m512d v1, __m512d v2) noexcept {
  // Create index vectors for interleaving
  __m512d interleave_v1 = _mm512_unpacklo_pd(v1, v2); // {a1, b1, a2, b2, a3, b3, a4, b4}
  __m512d interleave_v2 = _mm512_unpackhi_pd(v1, v2); // {a5, b5, a6, b6, a7, b7, a8, b8}
  return {interleave_v1, interleave_v2};
}

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
        auto const bl1_size   = M[bl1].target_shape()[0];
        auto const bl2_size   = M[bl2].target_shape()[0];
        auto const M1 = M[bl1];
        auto const M2 = M[bl2];
        auto M4       = M4_iw_(bl1, bl2);
        for (auto iw1 : iw_mesh)
          for (auto iw2 : iw_mesh)
#pragma clang loop unroll_count(16)
            for (auto iw3 : iw_mesh) {
              const auto iw4        = iw1 + iw3 - iw2;
              const auto total_size = bl1_size * bl1_size * bl2_size * bl2_size;
              const auto remainder            = total_size % 8;
              const auto inv_bl1_size_cubed = 1 / (bl1_size * bl2_size * bl2_size);
              const auto inv_bl2_size_squared = 1 / (bl2_size * bl2_size);
              const auto inv_bl1_size = 1 / bl1_size;
              const auto inv_bl1_size_squared = 1 / (bl1_size * bl1_size);
              const auto bl12 = (bl1_size * bl2_size);
              for (auto index = 0; index < total_size - remainder; index += 8) {
                // Load complex numbers as pairs of doubles
                const auto li1    = index * inv_bl1_size_cubed;
                const auto jk1    = (index * inv_bl2_size_squared) % bl12;
                const auto lk1    = (index * inv_bl1_size) % bl12;
                const auto ji1    = index * inv_bl1_size_squared;
                const auto index2 = index + 4;
                const auto li2    = index2 * inv_bl1_size_cubed;
                const auto jk2    = (index2 * inv_bl2_size_squared) % bl12;
                const auto lk2    = (index2 * inv_bl1_size) % bl12;
                const auto ji2    = index2 * inv_bl1_size_squared;
                // Separate real and imaginary parts
                const auto [M1val_real_v, M1val_imag_v] =
                   separate_real_imaginary(_mm512_loadu_pd(reinterpret_cast<double *>(M1[iw2.value(), iw1].data() + ji1)),
                                           _mm512_loadu_pd(reinterpret_cast<double *>(M1[iw2.value(), iw1].data() + ji2)));
                const auto [M2_lk_real_v, M2_lk_imag_v] =
                   separate_real_imaginary(_mm512_loadu_pd(reinterpret_cast<double *>(M2[iw4, iw3].data() + lk1)),
                                           _mm512_loadu_pd(reinterpret_cast<double *>(M2[iw4, iw3].data() + lk2)));
                const auto [M1_li_real_v, M1_li_imag_v] =
                   separate_real_imaginary(_mm512_loadu_pd(reinterpret_cast<double *>(M1[iw4, iw1].data() + li1)),
                                           _mm512_loadu_pd(reinterpret_cast<double *>(M1[iw4, iw1].data() + li2)));

                const auto [M2_jk_real_v, M2_jk_imag_v] =
                   separate_real_imaginary(_mm512_loadu_pd(reinterpret_cast<double *>(M2[iw2.value(), iw3].data() + jk1)),
                                           _mm512_loadu_pd(reinterpret_cast<double *>(M2[iw2.value(), iw3].data() + jk2)));
                const auto [M4_real_v, M4_imag_v] =
                   separate_real_imaginary(_mm512_loadu_pd(reinterpret_cast<double *>(M4[iw1, iw2, iw3].data() + index)),
                                           _mm512_loadu_pd(reinterpret_cast<double *>(M4[iw1, iw2, iw3].data() + index2)));
                // Perform the operations
                auto result_real_v = _mm512_add_pd(M4_real_v, _mm512_mul_pd(v_sign, _mm512_sub_pd(_mm512_mul_pd(M1val_real_v, M2_lk_real_v), _mm512_mul_pd(M1val_imag_v, M2_lk_imag_v))));
                auto result_imag_v = _mm512_add_pd(M4_imag_v, _mm512_mul_pd(v_sign, _mm512_add_pd(_mm512_mul_pd(M1val_real_v, M2_lk_imag_v), _mm512_mul_pd(M1val_imag_v, M2_lk_real_v))));
                if (bl1==bl2) [[unlikely]]{
                  result_real_v = _mm512_sub_pd(result_real_v, _mm512_mul_pd(v_sign, _mm512_sub_pd(_mm512_mul_pd(M1_li_real_v, M2_jk_real_v), _mm512_mul_pd(M1_li_imag_v, M2_jk_imag_v))));
                  result_imag_v = _mm512_sub_pd(result_imag_v, _mm512_mul_pd(v_sign, _mm512_add_pd(_mm512_mul_pd(M1_li_real_v, M2_jk_imag_v), _mm512_mul_pd(M1_li_imag_v, M2_jk_real_v))));
                }
                // Pack real and imaginary parts back together
                const auto [result_a, result_b] = interleave_vectors(result_real_v, result_imag_v);
                // Store the result back into M4
                _mm512_storeu_pd(reinterpret_cast<double *>(M4[iw1, iw2, iw3].data() + index), result_a);
                _mm512_storeu_pd(reinterpret_cast<double *>(M4[iw1, iw2, iw3].data() + index2), result_b);
              }
              if (remainder) [[unlikely]] {
                for (auto index = total_size - remainder; index < total_size; ++index) {
                const auto li = index / (bl1_size * bl2_size * bl2_size);
                const auto jk = (index / (bl2_size * bl2_size)) % (bl1_size * bl2_size);
                const auto lk = (index / bl1_size) % (bl2_size * bl2_size);
                const auto ji = index % (bl1_size * bl1_size);
                auto M1val    = M1[iw2.value(), iw1].data()[ji];
                M4[iw1, iw2, iw3].data()[index] += sign * M1val * M2[iw4, iw3].data()[lk];
                M4[iw1, iw2, iw3].data()[index] -= bl1 == bl2 ? sign * M1[iw4, iw1].data()[li] * M2[iw2.value(), iw3].data()[jk] : 0;
                }
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
