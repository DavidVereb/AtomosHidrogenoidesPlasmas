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
    P_final = lpmv(0, 0, x)
    P_foton = lpmv(q, 1, x)
    P_inicial = lpmv(m_i, l_i, x)

    integral_theta = np.sum(w * P_final * P_foton * P_inicial)
    integral_phi = 2*np.pi
    factor_angular += np.abs(integral_theta*integral_phi)**2

####################
### ESTADO FINAL ###
####################
E_f = datos["E_l0"][0]
u_f = datos["u_l0"][:, 0]

#########################
### ESTADOS INICIALES ###
#########################
n_max = 10 # Queremos que n=2,...,10
num_estados_i = min(n_max-l_i, len(datos["E_l1"]))
E_i = datos["E_l1"][:num_estados_i]
u_i = datos["u_l1"][:, :num_estados_i]

##############################################
### ELEMENTOS MATRIZ TRANSICIONES (RADIAL) ###
##############################################
# Elementos de matriz al cuadrado
S_if = np.array([])
for i in range(num_estados_i):
    integral_radial = np.trapezoid(np.conj(u_f)*r*u_i[:, i])
    S_if[i] = (integral_radial**2) * factor_angular

#####♯###########################
### FRECUENCIAS DE TRANSICIÓN ###
#################################
# A partir de ahora usamos el Sistema Internacional
nu_if = (E_i - E_f) * E_h / h

##############################
### COEFICIENTES EINSTEIN ###
#############################
A_if = 8*np.pi**2*(a_0*e)**2/(3*epsilon_0*hbar*c**3) * nu_if**3 * S_if

#######################
### PERFIL DE LINEA ###
#######################
n = np.arange(2, n_max+1)

sigma = 2*np.log(2)*np.sqrt(k*T_e/(m_p*c**2))*nu_if
gamma_L = 8*np.pi**2*n_e/(6*np.sqrt(3)) * (hbar/m_e)**2 * np.sqrt(2*m_e/(np.pi*k*T_e)) * (0.9-1.1/Z)*(3*n/(2*Z))**2*(n**2-3)

x, w = np.polynomial.hermite.hermgauss(40)
phi_V = np.array([])
for i in range(n_max):
    phi_V[i] = lambda nu: gamma_L[i]/(2*np.pi**(3/2))*np.sum(w/((nu-nu_if[i]-np.sqrt(2)*sigma[i]*x)**2 + 1/4*gamma_L[i]**2))

