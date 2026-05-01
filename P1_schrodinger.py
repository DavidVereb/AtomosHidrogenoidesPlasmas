"""
Importa la malla y los parámetros de config.py. Contruye el Hamiltoniano para cada valor de momento angular. Resuelve
los autovalores y normaliza las funciones de onda.

Exporta en "data/funciones_onda.npz" las energías E_nl y los vectores u_nl(r)
"""

from config import *

################################
### CONSTANTES CON DIMENSIÓN ###
################################
r_D_si = np.sqrt(epsilon_0*k*T_e/(n_e*e**2)) # Longitud de Debye
r_0_si = np.cbrt(3/(4*np.pi*n_e)) # Radio de la esfera de neutralidad

#################################
### CONSTANTES ADIMENSIONALES ###
#################################
r_D = r_D_si/ a_0 # Longitud de Debye adimensional
kT_e = k*T_e/E_h # ¿Energía térmica electrónica? Adimensional
r_0 = r_0_si/a_0 # Radio de la esfera de neutralidad adimensional
Gamma = Z**2/(r_0*kT_e) # Parámetro de acoplamiento adimensional

##############################
### POTENCIAL ADIMENSIONAL ###
##############################
V = lambda _r: -Z/_r*np.exp(-_r/r_D)*(1 - Gamma + 1/2*Gamma*_r/r_D)

####################
### HAMILTONIANO ###
####################
diagonal_principal = lambda _l: 1/dr**2 + V(r_inner) + _l*(_l+1)/(2*r_inner**2)
diagonal_secundaria = -1/(2*dr**2)*np.ones(N-2)

################
### ESPECTRO ###
################
limite_inferior = -0.5*Z**2 - 0.5

E_dict = {}
u_dict = {}

for l in range(4):
    E, u_inner = la.eigh_tridiagonal(diagonal_principal(l), diagonal_secundaria, select='v', select_range=(limite_inferior, -1e-6))

    u = np.zeros((N+1,  len(E)))
    u[1:-1, :] = u_inner

    E_dict[l] = E
    u_dict[l] = u

#####################
### NORMALIZACIÓN ###
#####################
for l in range(4):
    norma = np.sqrt(np.trapezoid(np.abs(u_dict[l])**2, x=r, axis=0))
    u_dict[l] /= norma


################
### GUARDADO ###
################
np.savez_compressed(
    'data/funciones_onda.npz',
    E_l0=E_dict[0], E_l1=E_dict[1], E_l2=E_dict[2], E_l3=E_dict[3],
    u_l0=u_dict[0], u_l1=u_dict[1], u_l2=u_dict[2], u_l3=u_dict[3]
)
