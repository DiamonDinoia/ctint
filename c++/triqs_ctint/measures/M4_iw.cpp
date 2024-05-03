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
    for (auto const bl1 : range(params.n_blocks())) {
      for (auto const bl2 : range(params.n_blocks())) {
        const auto bl1_size = M[bl1].target_shape()[0];
        const auto bl2_size = M[bl2].target_shape()[0];
        auto const M1       = M[bl1];
        auto const M2       = M[bl2];
        auto const M4       = M4_iw_(bl1, bl2);
        for (const auto iw1 : iw_mesh) {
          for (const auto iw2 : iw_mesh) {
            for (const auto iw3 : iw_mesh) {
              const auto iw4           = iw1 + iw3 - iw2;
              const auto total_size    = bl1_size * bl1_size * bl2_size * bl2_size;
              const auto bl2_size2     = bl2_size * bl2_size;
              const auto inv_bl2_size2 = 1 / bl2_size2;
              const auto remainder     = total_size % 2;

              // vectorize over the total size using intrinsics
              for (int index = 0; index < total_size - remainder; index += 2) { // Note: AVX2 can handle 2 complex doubles at a time
                const auto ij     = index * inv_bl2_size2;
                const auto kl     = index % bl2_size2;
                const auto M1val  = M1[iw2.value(), iw1].data()[ij];
                __m256d M1val_vec = _mm256_loadu_pd((double *)(M1[iw2.value(), iw1].data() + ij));
                __m256d M2_vec    = _mm256_loadu_pd((double *)(M2[iw4, iw3].data() + kl));
                // Multiply M1val and M2
                __m256d M1val_real = _mm256_permute_pd(M1val_vec, 0x0); // Real parts of M1val
                __m256d M1val_imag = _mm256_permute_pd(M1val_vec, 0xF); // Imaginary parts of M1val
                __m256d M2_real    = _mm256_permute_pd(M2_vec, 0x0);    // Real parts of M2
                __m256d M2_imag    = _mm256_permute_pd(M2_vec, 0xF);    // Imaginary parts of M2

                __m256d real = _mm256_sub_pd(_mm256_mul_pd(M1val_real, M2_real), _mm256_mul_pd(M1val_imag, M2_imag));
                __m256d imag = _mm256_add_pd(_mm256_mul_pd(M1val_real, M2_imag), _mm256_mul_pd(M1val_imag, M2_real));

                // Multiply by sign
                __m256d sign_vec = _mm256_set1_pd(sign);
                real             = _mm256_mul_pd(real, sign_vec);
                imag             = _mm256_mul_pd(imag, sign_vec);

                // Add to M4
                __m256d M4_vec  = _mm256_loadu_pd((double *)(M4[iw1, iw2, iw3].data() + index));
                __m256d M4_real = _mm256_permute_pd(M4_vec, 0x0); // Real parts of M4
                __m256d M4_imag = _mm256_permute_pd(M4_vec, 0xF); // Imaginary parts of M4

                M4_real = _mm256_add_pd(M4_real, real);
                M4_imag = _mm256_add_pd(M4_imag, imag);

                // Store the result back into M4
                __m256d result = _mm256_unpacklo_pd(M4_real, M4_imag); // Interleave real and imaginary parts
                _mm256_storeu_pd((double *)(M4[iw1, iw2, iw3].data() + index), result);
              }
              for (int index = total_size - remainder; index < total_size; index++) {
                const auto ij = index * inv_bl2_size2;
                const auto kl = index % bl2_size2;
                M4[iw1, iw2, iw3].data()[index] += sign * M1[iw2.value(), iw1].data()[ij] * M2[iw4, iw3].data()[kl];
              }
            }
          }
        }
      }
    }

    for (auto const bl1 : range(params.n_blocks())) {
      const auto bl1_size = M[bl1].target_shape()[0];
      const auto bl2_size = M[bl1].target_shape()[0];
      auto const M1       = M[bl1];
      auto const M2       = M[bl1];
      auto const M4       = M4_iw_(bl1, bl1);
      for (const auto iw1 : iw_mesh) {
        for (const auto iw2 : iw_mesh) {
          for (const auto iw3 : iw_mesh) {
            const auto iw4 = iw1 + iw3 - iw2;
            for (const auto i : range(bl1_size)) {
              for (const auto j : range(bl1_size)) {
                for (const auto k : range(bl2_size)) {
                  for (const auto l : range(bl2_size)) { M4[iw1, iw2, iw3](i, j, k, l) -= sign * M1[iw4, iw1](l, i) * M2[iw2.value(), iw3](j, k); }
                }
              }
            }
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
