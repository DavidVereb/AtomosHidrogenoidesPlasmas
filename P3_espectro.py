"""
Carga "data/coeficientes_einstein.npz" y los parámetros de config.py. Resuelve el sistema de Saha-Bolzmann para sacar
las poblaciones n_Hnl. Construye el perfil de Voigt para cada línea espectral combinando los ensanchamientos Doppler y
colisional. Finalmente suma todo para calcular la emisividad j(v) y la intensidad específica I(v).

Exporta en "data/espectro_final.npz" el vector de frecuencias v y el vector de intensidad específica I(v) de la serie
Lyman.
"""

from config import *

######################
### CARGA DE DATOS ###
######################
datos = np.load("data/coeficientes_einstein.npz")

#######################
### PERFIL DE LINEA ###
#######################
n_i = datos["n_i"]
nu_if = datos["nu_if"]

sigma = 2*np.log(2)*np.sqrt(k*T_e/(m_p*c**2))*nu_if
gamma_L = 8*np.pi**2*n_e/(6*np.sqrt(3)) * (hbar/m_e)**2 * np.sqrt(2*m_e/(np.pi*k*T_e)) * (0.9-1.1/Z)*(3*n_i/(2*Z))**2*(n_i**2-3)

x, w = np.polynomial.hermite.hermgauss(40)
phi_V = np.zeros(len(n_i))
for i in range(len(n_i)):
    phi_V[i] = lambda nu: gamma_L[i]/(2*np.pi**(3/2))*np.sum(w/((nu-nu_if[i]-np.sqrt(2)*sigma[i]*x)**2 + 1/4*gamma_L[i]**2))

