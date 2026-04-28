"""
Constantes físicas, definición de malla espacial, parámetros del plasma.
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.special import lpmv
import scipy.constants as cte
from scipy.constants import k, epsilon_0, e, h, m_e, c, hbar

E_h, _, _ = cte.physical_constants['Hartree energy']
a_0, _, _ = cte.physical_constants['Bohr radius']

#########################
### PARÁMETROS PLASMA ###
#########################

T_e = 11600
n_e = 1e18
Z = 1

##############################
### DISCRETIZACIÓN ATÓMICA ###
##############################
R_max = 350
N = 200000
r = np.linspace(0, R_max, N+1)
dr = r[1] - r[0]
r_inner = r[1:-1]
