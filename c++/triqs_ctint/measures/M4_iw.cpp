#include "./M4_iw.hpp"
#include "./intrinsics.h"

namespace triqs_ctint::measures {

  M4_iw::M4_iw(params_t const &params_, qmc_config_t const &qmc_config_, container_set *results)
     : params(params_), qmc_config(qmc_config_), buf_arrarr(params_.n_blocks()) {

    // Construct Matsubara mesh
    mesh::imfreq iw_mesh{params.beta, Fermion, params.n_iw_M4};
    auto M4_iw_mesh = iw_mesh * iw_mesh * iw_mesh;

    // Init measurement container and capture view
    results->M4_iw = make_block2_gf(M4_iw_mesh, params.gf_struct);
    M4_iw_.rebind(results->M4_iw.value());
    M4_iw_() = 0;

    // Construct Matsubara mesh for temporary Matrix
    mesh::imfreq iw_mesh_large{params.beta, Fermion, 3 * params.n_iw_M4};
    auto M_mesh = iw_mesh_large * iw_mesh;

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

  // function to accumulate the intermediate scattering matrix
  // it uses the SIMD intrinsics to vectorize the loops
  // bl1_batch and bl2_batch are the batch sizes for the first and second block respectively that determine the width
  // of the SIMD instructions
  // if a wider SIMD than supported is used it will fall back to the smaller version
  template <unsigned bl1_batch, unsigned bl2_batch> void M4_iw::accumulate(mc_weight_t sign, unsigned bl1, unsigned bl2) {
    // if the user requests a SIMD size larger than the supported one, it will fall back to the supported one
    static auto constexpr simd1_size = std::min(bl1_batch, widest_simd<double>());
    static auto constexpr simd2_size = std::min(bl2_batch, widest_simd<double>());
    auto const &iw_mesh = std::get<0>(M4_iw_(0, 0).mesh());
    auto const bl1_size = M[bl1].target_shape()[0];
    auto const bl2_size = M[bl2].target_shape()[0];
    auto const M1       = M[bl1];
    auto const M2       = M[bl2];
    auto M4             = M4_iw_(bl1, bl2);
    for (const auto &iw1 : iw_mesh) {
      for (const auto &iw2 : iw_mesh) {
        for (const auto &iw3 : iw_mesh) {
          const auto iw4 = iw1 + iw3 - iw2;
          for (auto i : range(bl1_size)) {
            for (auto j : range(bl1_size)) {
              const auto M1val = M1[iw2.value(), iw1](j, i) * sign;
// USE_INTRINSICS is both a flag and a parameter because the block would not compile if the code is invalid inside the block
              if constexpr (USE_INTRINSICS && bl1_batch>1) {
#if USE_INTRINSICS == 1
                const auto bl2square            = bl2_size * bl2_size;
                using Type                      = decltype(M1val.real());
                const auto truncated_size       = bl2square & (-simd1_size);
                // AND-ing with -simd1_size will truncate the size to the nearest multiple of the SIMD instruction
                // this works only if the size of the simd1_size instruction is a power of 2
                const auto [M1s_real, M1s_imag] = set_vector_to_complex<Vec<Type, simd1_size>>(M1val);
                uint64_t index                  = 0;
                for (; index < truncated_size; index += simd1_size) {
                  auto *const RESTRICT m4_ptr               = &M4[iw1, iw2, iw3](i, j, 0, 0) + index;
                  const auto [M4v1, M4v2]                   = load<Type, simd1_size>(m4_ptr);
                  const auto [real, imag]                   = load_and_separate<Type, simd1_size>(M2[iw4, iw3].data() + index);
                  const auto [real_res, imag_res]           = complex_mul(M1s_real, M1s_imag, real, imag);
                  const auto [interleave_v1, interleave_v2] = interleave_vectors(real_res, imag_res);
                  store(m4_ptr, M4v1 + interleave_v1);
                  store(m4_ptr + (simd1_size / 2), M4v2 + interleave_v2);
                }
                for (; index < bl2square; index++) { (&M4[iw1, iw2, iw3](i, j, 0, 0))[index] += M1val * M2[iw4, iw3].data()[index]; }
#endif
              } else {
                for (auto index : range(bl2_size * bl2_size)) { (&M4[iw1, iw2, iw3](i, j, 0, 0))[index] += M1val * M2[iw4, iw3].data()[index]; }
              }
              if (bl1 == bl2) [[unlikely]] {
                for (const auto k : range(bl2_size)) {
                  const auto M2sval = M2[iw2.value(), iw3](j, k) * sign;
                  if constexpr (USE_INTRINSICS==1 && bl2_batch>1) {
#if USE_INTRINSICS == 1
                    using Type                = decltype(M2sval.real());
                    const auto truncated_size = bl2_size & (-simd2_size);
                    // see above for explanation
                    const auto [M2s_real, M2s_imag] = set_vector_to_complex<Vec<Type, simd2_size>>(M2sval);
                    uint64_t index                  = 0;
                    for (; index < truncated_size; index += simd2_size) {
                      const auto [real, imag]                   = load_and_separate<Type, simd2_size>(&M1[iw4, iw1](index, i));
                      const auto [real_res, imag_res]           = complex_mul(M2s_real, M2s_imag, real, imag);
                      const auto [interleave_v1, interleave_v2] = interleave_vectors(real_res, imag_res);
                      auto *const RESTRICT m4_ptr               = &M4[iw1, iw2, iw3](i, j, k, index);
                      const auto [M4v1, M4v2]                   = load<Type, simd2_size>(m4_ptr);
                      store(m4_ptr, M4v1 - interleave_v1);
                      store(m4_ptr + (simd2_size / 2), M4v2 - interleave_v2);
                    }
                    for (; index < bl2_size; index++) { M4[iw1, iw2, iw3](i, j, k, index) -= M2sval * M1[iw4, iw1](index, i); }
#endif
                  }else {
                    for (const auto l : range(bl2_size)) { M4[iw1, iw2, iw3](i, j, k, l) -= M2sval * M1[iw4, iw1](l, i); }
                  }
                }
              }
            }
          }
        }
      }
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
               });
    for (auto &buf_arr : buf_arrarr)
      for (auto &buf : buf_arr) buf.flush(); // Flush remaining points from all buffers

    for (const int bl1 : range(params.n_blocks())) { // FIXME c++17 Loops
      for (const int bl2 : range(params.n_blocks())) {
        auto const bl2_size = M[bl2].target_shape()[0];
        // Dispatch to the correct SIMD instruction width based on the size of the blocks
        // It will try to use the widest SIMD instruction available for the given block sizes
        // TODO: fold expressions might be an option to simplify the code
        if constexpr (USE_INTRINSICS==1) {
          if (bl2_size >= 16) {
            accumulate<8, 8>(sign, bl1, bl2);
          } else if (bl2_size >= 8) {
            accumulate<8, 4>(sign, bl1, bl2);
          } else if (bl2_size >= 4) {
            accumulate<8, 2>(sign, bl1, bl2);
          } else if (bl2_size >= 3) {
            accumulate<4, 1>(sign, bl1, bl2);
          } else if (bl2_size >= 2) {
            accumulate<2, 1>(sign, bl1, bl2);
          } else {
            accumulate<1, 1>(sign, bl1, bl2);
          }
        } else {
          accumulate<1, 1>(sign, bl1, bl2);
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
