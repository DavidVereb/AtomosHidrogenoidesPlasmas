"""
Carga "data/funciones_onda.npz". Evalúa las integrales radiales y la parte angular usando polinomios de Legendre para
calcular los elementos de matriz de las transiciones dipolares. Aplica las reglas de selección para descartar integrales
que den cero. Convierte los resultados a unidades del S.I. y calcula los coeficientes de Einstein Aif.

Exporta en "data/coeficientes_einstein.npz" las frecuencias de transición v_if y los coeficientes A_if.
"""
from scipy.constants import epsilon_0

from config import *

######################
### CARGA DE DATOS ###
######################
datos = np.load("data/funciones_onda.npz")

###############################################
### ELEMENTOS MATRIZ TRANSICIONES (ANGULAR) ###
###############################################
x, w = np.polynomial.legendre.leggauss(20)

# Estamos estudiando serie Lyman (n_f = 1)
l_f = 0
m_f = 0

# Regla de selección
l_i = 1

factor_angular = 0.0

for q in [-1, 0, 1]:
    # Regla de selección
    m_i = -q

    # Polinomios de Legendre
    P_final = lpmv(m_f, l_f, x)
    P_foton = lpmv(q, 1, x)
    P_inicial = lpmv(m_i, l_i, x)

    integral_theta = np.sum(w * P_final * P_foton * P_inicial)
    integral_phi = 2*np.pi
    factor_angular += np.abs(integral_theta*integral_phi)**2

####################
### ESTADO FINAL ###
####################
E_f = datos["E_0"]
u_f = datos["u_0"]

#########################
### ESTADOS INICIALES ###
#########################
#n_max = 10 # Nos interesan n=2,...,10
#num_estados_i = min(n_max-l_i, len(datos["E_l1"]))
E_i = datos["E_l1"]
u_i = datos["u_l1"]

##############################################
### ELEMENTOS MATRIZ TRANSICIONES (RADIAL) ###
##############################################
# Elementos de matriz al cuadrado
S_if = np.zeros(len(E_i))
for i in range(len(E_i)):
    integral_radial = np.trapezoid(np.conj(u_f)*r*u_i[:, i], x=r)
    S_if[i] = (integral_radial**2) * factor_angular

#####♯###########################
### FRECUENCIAS DE TRANSICIÓN ###
#################################
# A partir de ahora usamos el Sistema Internacional
nu_if = (E_i - E_f) * E_h / h

#############################
### COEFICIENTES EINSTEIN ###
#############################
A_if = 8*np.pi**2*(a_0*e)**2/(3*epsilon_0*hbar*c**3) * nu_if**3 * S_if

################
### GUARDADO ###
################
np.savez_compressed(
    'data/coeficientes_einstein.npz',
    nu_if = nu_if,
    A_if = A_if,
)
