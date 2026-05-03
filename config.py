"""
Constantes físicas, definición de malla espacial, parámetros del plasma.
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.special import lpmv
from scipy.special import voigt_profile # Prueba temporal
import scipy.constants as cte
from scipy.constants import k, epsilon_0, e, h, m_e, m_p, c, hbar

E_h, _, _ = cte.physical_constants['Hartree energy']
a_0, _, _ = cte.physical_constants['Bohr radius']

#########################
### PARÁMETROS PLASMA ###
#########################

#T_e = 11_600
T_e = 2_000_000
n_e = 1e14
Z = 1
L = 1

##############################
### DISCRETIZACIÓN ATÓMICA ###
##############################
R_max = 350
N = 200_000
r = np.linspace(0, R_max, N+1)
dr = r[1] - r[0]
r_inner = r[1:-1]

#################################
### DISCRETIZACIÓN FRECUENCIA ###
#################################
N_nu = 100_000