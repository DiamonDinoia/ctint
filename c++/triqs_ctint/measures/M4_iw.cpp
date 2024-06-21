#include <xsimd/xsimd.hpp>

#include "./M4_iw.hpp"
#include "./intrinsics.h"

namespace triqs_ctint::measures {

  template <class T, auto N> static constexpr auto min_simd_width() {
    // finds the smallest simd width that can handle N elements
    // simd size is batch size the SIMD width in xsimd terminology
    if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
      return min_simd_width<T, N * 2>();
    } else {
      return N;
    }
  };

  template <class T, auto N> static constexpr auto max_simd_width() {
    // finds the smallest simd width that can handle N elements
    // simd size is batch size the SIMD width in xsimd terminology
    if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
      return max_simd_width<T, N / 2>();
    } else {
      return N;
    }
  };

  static constexpr auto min_width = min_simd_width<double, 2>();
  static constexpr auto max_width = max_simd_width<double, 64>();

  static_assert(!std::is_void_v<xsimd::make_sized_batch_t<double, min_width>>, "min_width is not valid");
  static_assert(!std::is_void_v<xsimd::make_sized_batch_t<double, max_width>>, "max_width is not valid");

  template <auto N, class T = double> static constexpr auto get_arch_for_size() { return typename xsimd::make_sized_batch_t<T, N>::arch_type{}; }

  template <class T, auto N> using make_complex_sized_batch_t = xsimd::batch<std::complex<T>, decltype(get_arch_for_size<N, T>())>;

  static_assert(!std::is_void_v<make_complex_sized_batch_t<double, min_width>>, "failed to create min_width complex batch");
  static_assert(!std::is_void_v<make_complex_sized_batch_t<double, max_width>>, "failed to create max_width complex batch");

  // function to accumulate the intermediate scattering matrix
  // it uses the SIMD intrinsics to vectorize the loops
  // bl1_batch and bl2_batch are the batch sizes for the first and second block respectively that determine the width
  // of the SIMD instructions
  // if a wider SIMD than supported is used it will fall back to the smaller version

  template <auto bl1_batch, auto bl2_batch> void M4_iw::accumulate(mc_weight_t sign, auto bl1, auto bl2) {
    // if the user requests a SIMD size larger than the supported one, it will fall back to the supported one
    static auto constexpr simd1_size = std::max(std::min(bl1_batch, max_width), min_width);
    static auto constexpr simd2_size = std::max(std::min(bl2_batch, max_width), min_width);
    using batch1_t                   = make_complex_sized_batch_t<double, simd1_size>;
    using batch2_t                   = make_complex_sized_batch_t<double, simd2_size>;
    auto const &iw_mesh              = std::get<0>(M4_iw_(0, 0).mesh());
    auto const bl1_size              = M[bl1].target_shape()[0];
    auto const bl2_size              = M[bl2].target_shape()[0];
    auto const M1                    = M[bl1];
    auto const M2                    = M[bl2];
    auto M4                          = M4_iw_(bl1, bl2);
    for (const auto &iw1 : iw_mesh) {
      for (const auto &iw2 : iw_mesh) {
        for (const auto &iw3 : iw_mesh) {
          const auto iw4 = iw1 + iw3 - iw2;
          for (auto i : range(bl1_size)) {
            for (auto j : range(bl1_size)) {
              uint64_t index = 0;

              const auto M1val     = M1[iw2.value(), iw1](j, i) * sign;
              const auto bl2square = bl2_size * bl2_size;
              if constexpr (bl1_batch >= min_width) {
                const auto M1_v           = batch1_t(M1val);
                const auto truncated_size = bl2square & (-batch1_t::size);
                for (; index < truncated_size; index += batch1_t::size) {
                  auto *const RESTRICT m4_ptr = &M4[iw1, iw2, iw3](i, j, 0, 0) + index;
                  const auto batch            = batch1_t::load_unaligned(m4_ptr);
                  const auto M2_batch         = batch1_t::load_unaligned(M2[iw4, iw3].data() + index);
                  const auto result           = xsimd::fma(M1_v, M2_batch, batch);
                  result.store_unaligned(m4_ptr);
                }
              }
              for (; index < bl2square; index++) {
                (&M4[iw1, iw2, iw3](i, j, 0, 0))[index] = xsimd::fma(M1val, M2[iw4, iw3].data()[index], (&M4[iw1, iw2, iw3](i, j, 0, 0))[index]);
              }

              if (bl1 == bl2) [[unlikely]] {
                for (const auto k : range(bl2_size)) {
                  index             = 0;
                  const auto M2sval = M2[iw2.value(), iw3](j, k) * sign;
                  if constexpr (bl2_batch >= min_width) {
                    const auto M2s_v          = batch2_t(M2sval);
                    const auto truncated_size = bl2_size & (-batch2_t::size);
                    for (; index < truncated_size; index += batch2_t::size) {
                      auto *const RESTRICT m4_ptr = &M4[iw1, iw2, iw3](i, j, k, index);
                      const auto batch            = batch2_t::load_unaligned(m4_ptr);
                      const auto M1_batch         = batch2_t::load_unaligned(&M1[iw4, iw1](index, i));
                      // fnma(x, y, z) -> -(x*y) + z
                      const auto result = xsimd::fnma(M2s_v, M1_batch, batch);
                      result.store_unaligned(m4_ptr);
                    }
                  }
                  for (; index < bl2_size; index++) {
                    M4[iw1, iw2, iw3](i, j, k, index) = xsimd::fnma(M2sval, M1[iw4, iw1](index, i), M4[iw1, iw2, iw3](i, j, k, index));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

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

    for (int bl1 : range(params.n_blocks())) { // FIXME c++17 Loops
      for (int bl2 : range(params.n_blocks())) {
        auto const bl2_size = M[bl2].target_shape()[0];
        // Dispatch to the correct SIMD instruction width based on the size of the blocks
        // It will try to use the widest SIMD instruction available for the given block sizes
        // TODO: fold expressions might be an option to simplify the code
        if (bl2_size >= 8) {
          accumulate<8, 8>(sign, bl1, bl2);
        } else if (bl2_size >= 4) {
          accumulate<8, 4>(sign, bl1, bl2);
        } else if (bl2_size >= 3) {
          accumulate<8, 2>(sign, bl1, bl2);
        } else if (bl2_size >= 2) {
          accumulate<4, 2>(sign, bl1, bl2);
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
