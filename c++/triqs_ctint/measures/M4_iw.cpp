#include "./M4_iw.hpp"
#include <cmath>

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
    // use chrono to measure time
    const auto start = std::chrono::high_resolution_clock::now();
    for (auto const bl1 : range(params.n_blocks())) // FIXME c++17 Loops
      for (auto const bl2 : range(params.n_blocks())) {
        const auto bl1_size = M[bl1].target_shape()[0];
        const auto bl2_size = M[bl2].target_shape()[0];
        auto const M1      = M[bl1];
        auto const M2      = M[bl2];
        auto const M4       = M4_iw_(bl1, bl2);
        for (const auto iw1 : iw_mesh)
          for (const auto iw2 : iw_mesh)
            for (const auto iw3 : iw_mesh) {
              const auto iw4 = iw1 + iw3 - iw2;
              for (const auto i : range(bl1_size))
                for (const auto j : range(bl1_size)) {
                  // flatten left side expr here
                  const auto M1val = M1[iw2.value(), iw1](j, i);
                  for (const auto k : range(bl2_size)) {
                    const auto M2val = M2[iw2.value(), iw3](j, k);
                    for (const auto l : range(bl2_size)) {
                      M4[iw1, iw2, iw3](i, j, k, l) +=
                         sign * (M1val * M2[iw4, iw3](l, k) - (__builtin_expect(bl1 == bl2, false) ? M1[iw4, iw1](l, i) * M2val : 0));
                    }
                  }
                }
            }
      }
    const auto end                        = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "M4_iw: " << elapsed.count() << " seconds" << std::endl;
  }

  void M4_iw::collect_results(mpi::communicator const &comm) {
    // Collect results and normalize
    Z      = mpi::all_reduce(Z, comm);
    M4_iw_ = mpi::all_reduce(M4_iw_, comm);
    M4_iw_ = M4_iw_ / (Z * params.beta);
  }

} // namespace triqs_ctint::measures
