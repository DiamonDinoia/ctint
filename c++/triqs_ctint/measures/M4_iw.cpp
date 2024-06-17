#include <xsimd/xsimd.hpp>

#include "./M4_iw.hpp"
#include "./intrinsics.h"

namespace triqs_ctint::measures {

  using batch_t = xsimd::batch<std::complex<double>>;

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
               });
    for (auto &buf_arr : buf_arrarr)
      for (auto &buf : buf_arr) buf.flush(); // Flush remaining points from all buffers

    for (int bl1 : range(params.n_blocks())) { // FIXME c++17 Loops
      for (int bl2 : range(params.n_blocks())) {
        auto const &iw_mesh = std::get<0>(M4_iw_(0, 0).mesh());
        auto const bl1_size = M[bl1].target_shape()[0];
        auto const bl2_size = M[bl2].target_shape()[0];
        auto const M1       = M[bl1];
        auto const M2       = M[bl2];
        auto M4             = M4_iw_(bl1, bl2);

        for (auto iw1 : iw_mesh) {
          for (auto iw2 : iw_mesh) {
            for (auto iw3 : iw_mesh) {
              const auto iw4 = iw1 + iw3 - iw2;
              for (int i : range(bl1_size)) {
                for (auto j : range(bl1_size)) {
                  uint64_t index = 0;
                  {
                    const auto M1val          = M1[iw2.value(), iw1](j, i) * sign;
                    const auto bl2square      = bl2_size * bl2_size;
                    const auto M1_v           = batch_t(M1val);
                    const auto truncated_size = bl2square & (-batch_t::size);
                    if (bl2square > batch_t::size) {
                      for (; index < truncated_size; index += batch_t::size) {
                        auto *const RESTRICT m4_ptr = &M4[iw1, iw2, iw3](i, j, 0, 0) + index;
                        const auto batch            = batch_t::load_unaligned(m4_ptr);
                        const auto M2_batch         = batch_t::load_unaligned(M2[iw4, iw3].data() + index);
                        const auto result           = xsimd::fma(M1_v, M2_batch, batch);
                        result.store_unaligned(m4_ptr);
                      }
                    }
                    for (; index < bl2square; index++) { (&M4[iw1, iw2, iw3](i, j, 0, 0))[index] += M1val * M2[iw4, iw3].data()[index]; }
                  }
                  if (bl1 == bl2) [[unlikely]] {
                    for (const auto k : range(bl2_size)) {
                      index                     = 0;
                      const auto M2sval = M2[iw2.value(), iw3](j, k) * sign;
                      const auto M2s_v          = batch_t(M2sval);
                      const auto truncated_size = bl2_size & (-batch_t::size);
                      if (bl2_size > batch_t::size) {
                        for (; index < truncated_size; index += batch_t::size) {
                          auto *const RESTRICT m4_ptr = &M4[iw1, iw2, iw3](i, j, k, index);
                          const auto batch            = batch_t::load_unaligned(m4_ptr);
                          const auto M1_batch         = batch_t::load_unaligned(&M1[iw4, iw1](index, i));
                          const auto result           = xsimd::fms(M2s_v, M1_batch, batch);
                          result.store_unaligned(m4_ptr);
                        }
                      }
                      for (; index < bl2_size; index++) { M4[iw1, iw2, iw3](i, j, k, index) -= M2sval * M1[iw4, iw1](index, i); }
                    }
                  }
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
