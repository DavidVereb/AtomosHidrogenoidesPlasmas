"""
Carga "data/funciones_onda.npz". Evalúa las integrales radiales y la parte angular usando polinomios de Legendre para
calcular los elementos de matriz de las transiciones dipolares. Aplica las reglas de selección para descartar integrales
que den cero. Convierte los resultados a unidades del S.I. y calcula los coeficientes de Einstein Aif.

Exporta en "data/coeficientes_einstein.npz" las frecuencias de transición v_if y los coeficientes A_if.
"""

from config import *

######################
### CARGA DE DATOS ###
######################
datos = np.load("data/funciones_onda.npz")

###############
### ESTADOS ###
###############
E_f = datos["E_l0"][0]
u_f = datos["u_l0"][:, 0]

E_i = datos["E_l1"]
u_i = datos["u_l1"]

# A partir de ahora todo es SI
v_if  = (E_f-E_i)*E_h*e/h

N_Gauss = 20

x, w = np.polynomial.legendre.leggauss(N_Gauss)

I = 0

for q in [-1, 0, 1]:
    I += np.sum(w*lpmv(0, 0,x)*lpmv(q, 1, x)*lpmv(-q, 1,x))

I *= 2*np.pi

gamma_G = np.sqrt(2*np.log(2))*np.sqrt(k*T_e/(m_e*c**2))*v_if
sigma = np.sqrt(2*np.log(2))*gamma_G

gamma_L = 8*np.pi**2*n_e/(6*np.sqrt(3))*hbar**2/m_e**2*np.sqrt(2*m_e/(np.pi*k*T_e))*(0.9-1.1/Z)*(3*)