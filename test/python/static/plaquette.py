from triqs_ctint import Solver

from itertools import product
import triqs.utility.mpi as mpi
from triqs.gf import *
from h5 import *
from triqs.operators import c, c_dag, n
from triqs.utility.h5diff import assert_gfs_are_close, h5diff
from triqs.operators.util.hamiltonians import h_int_kanamori
from numpy import matrix, array

test_name = "plaquette"

# --------- Physical Parameters ----------

U = 1.0  # Density-density interaction
t = 1.0  # Hopping
mu = U / 2.0  # Chemical Potential
beta = 100.0  # Inverse temperature

# --------- Define hopping matrix and interaction hamiltonian ----------

hloc0_mat = -array(
    [
        [mu, t, 0, t],  #
        [t, mu, t, 0],  #
        [0, t, mu, t],  #
        [t, 0, t, mu],  #
    ]
)
h_int = sum(U * n("up", i) * n("dn", i) for i in range(4))

# --------- set up block structure ---------

block_names = ["dn", "up"]
gf_struct = [(bl, 4) for bl in block_names]

# --------- Construct the ctint solver ----------
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=200, n_tau=10001)

# --------- Initialize the non-interacting Green's function ----------
for bl, g_bl in S.G0_iw:
    g_bl << inverse(iOmega_n - hloc0_mat)

# --------- Solve! ----------
S.solve(
    h_int=h_int,
    n_cycles=10,
    length_cycle=100,
    n_s=2,
    alpha_mode='trivial',
    n_warmup_cycles=1000,
    measure_M4_iw=True,
    n_iw_M4=32,
    # nfft_buf_size=100000,
    # measure_M3pp_tau=True,
    # measure_M3ph_tau=True,
    # measure_M3xph_tau=True,
    # n_iw_M3=10,
    # n_iW_M3=10,
    # n_tau_M3=41,
    # measure_chi2pp_tau=True,
    # measure_chi2ph_tau=True,
    # n_iw_chi2=10,
    # n_tau_chi2=21,
    # measure_chiAB_tau=True,
    # chi_A_vec=[n("up", 0) + n("dn", 0)],
    # chi_B_vec=[n("up", 0) + n("dn", 0)],
    # post_process=True,
    post_process=False,
)

# -------- Save in archive ---------
with HDFArchive("%s.ref.h5" % test_name, "r") as arch:
    assert_gfs_are_close(S.M4_iw['up','dn'][0,0,0,0], arch["M4_iw_ud_0000"], 1e-12)
    assert_gfs_are_close(S.M4_iw['up','dn'][0,1,2,3], arch["M4_iw_ud_0123"], 1e-12)
    assert_gfs_are_close(S.M4_iw['up','up'][0,0,0,0], arch["M4_iw_uu_0000"], 1e-12)
    assert_gfs_are_close(S.M4_iw['up','up'][0,1,2,3], arch["M4_iw_uu_0123"], 1e-12)

# with HDFArchive("%s.out.h5" % test_name, "w") as arch:
    # arch["M4_iw_ud_0000"] = S.M4_iw['up','dn'][0,0,0,0]
    # arch["M4_iw_ud_0123"] = S.M4_iw['up','dn'][0,1,2,3]
    # arch["M4_iw_uu_0000"] = S.M4_iw['up','up'][0,0,0,0]
    # arch["M4_iw_uu_0123"] = S.M4_iw['up','up'][0,1,2,3]
    # arch["M4_iw"] = S.M4_iw
    # arch["G0_iw"] = S.G0_iw
    # arch["G_iw"] = S.G_iw
    # arch["G2_iw"] = S.G2_iw
    # arch["chi3pp_iw"] = S.chi3pp_iw
    # arch["chi3ph_iw"] = S.chi3ph_iw
    # arch["chi3xph_iw"] = S.chi3xph_iw
    # arch["chi2pp_iw"] = S.chi2pp_iw
    # arch["chi2ph_iw"] = S.chi2ph_iw
    # arch["chiAB_iw"] = S.chiAB_iw
    # arch["chi2pp_tau_from_M3"] = S.chi2pp_tau_from_M3
    # arch["chi2ph_tau_from_M3"] = S.chi2ph_tau_from_M3
    # arch["chi2xph_tau_from_M3"] = S.chi2xph_tau_from_M3

# -------- Compare ---------
# h5diff("%s.out.h5" % test_name, "%s.ref.h5" % test_name)
