from triqs_ctint import Solver

from itertools import product
import triqs.utility.mpi as mpi
from triqs.gf import *
from h5 import *
from triqs.operators import c, c_dag, n
from triqs.utility.h5diff import h5diff
from numpy import matrix, array

test_name = "plaquette"

######## physical parameters ########
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
S = Solver(beta = beta,
               gf_struct = gf_struct,
               n_iw = 100,
               n_tau = 201)

# --------- Initialize the non-interacting Green's function ----------
for bl, g_bl in S.G0_iw: g_bl << inverse(iOmega_n - hloc0_mat)

# --------- Solve! ----------
S.solve(h_int=h_int,
        n_cycles = 10,
        length_cycle = 50,
        n_warmup_cycles = 100,
        random_seed = 34788,
        measure_M4_iw = True,
        measure_M4pp_iw = True,
        measure_M4ph_iw = True,
        n_iw_M4 = 24,
        n_iW_M4 = 24,
        nfft_buf_size = 100000,
        post_process = False )
