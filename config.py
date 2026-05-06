"""
Constantes físicas, definición de malla espacial, parámetros del plasma.
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.special import lpmv
from scipy.special import voigt_profile # Prueba temporal
from scipy.signal import fftconvolve
import scipy.constants as cte
from scipy.constants import k, epsilon_0, e, h, m_e, m_p, c, hbar

E_h, _, _ = cte.physical_constants['Hartree energy']
a_0, _, _ = cte.physical_constants['Bohr radius']

#########################
### PARÁMETROS PLASMA ###
#########################

T_e_eV = 3 # eV
T_e = T_e_eV*11_600 # Kelvin
n_e_cm = 1e18
n_e = n_e_cm*1e6 # m^-3
Z = 1
L = 1

###############################
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
N_nu_base = 100_000
N_nu_local = 100_000

#############################
### OPCIONES DE EJECUCIÓN ###
#############################
voigt_manual = False

print(e/k)