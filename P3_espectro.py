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
datos_P1 = np.load("data/funciones_onda.npz")
datos_P2 = np.load("data/coeficientes_einstein.npz")
nu_if = datos_P2["nu_if"]
A_if = datos_P2["A_if"]

#########################
### MALLA FRECUENCIAS ###
#########################
nu_min = np.min(nu_if)*0.95
nu_max = np.max(nu_if)*1.05
nu = np.linspace(nu_min, nu_max, N_nu)

#######################
### PERFIL DE LINEA ###
#######################
sigma = 2*np.log(2)*np.sqrt(k*T_e/(m_p*c**2))*nu_if
n_i = np.arange(2, 2+len(nu_if))
gamma_L = 8*np.pi**2*n_e/(6*np.sqrt(3)) * (hbar/m_e)**2 * np.sqrt(2*m_e/(np.pi*k*T_e)) * (0.9-1.1/Z)*(3*n_i/(2*Z))**2*(n_i**2-3)
# ¿Guion puede estar mal?
gamma_L = np.abs(gamma_L)

"""
phi_V = []
for i in range(len(nu_if)):
    phi_G = 1/(sigma[i]*np.sqrt(2*np.pi))*np.exp(-(nu-nu_if[i])**2/(2*sigma[i]**2))
    phi_L = 1/np.pi*(gamma_L[i]/2)/((nu-np.mean(nu))** 2+(gamma_L[i]/2)**2)

    perfil = np.convolve(phi_G, phi_L, mode='same')
    phi_V.append(perfil)
"""

phi_V = []
for i in range(len(nu_if)):
    perfil = voigt_profile(nu-nu_if[i], sigma[i], gamma_L[i]/2)
    phi_V.append(perfil)


#########################
### FUNCIÓN PARTICIÓN ###
#########################
G_H = 2*np.exp(-datos_P1["E_0"]*E_h/(k*T_e)) # Incluimos el estado fundamental n=1, l=0
for l in range(4):
    E_l = datos_P1[f"E_l{l}"]
    g_l = 2*(2*l+1)
    # Pasamos de Hartrees a Julios
    for E_nl in E_l:
        G_H += g_l*np.exp(-E_nl*E_h/(k * T_e))

#######################################
### ENERGÍA POTENCIAL DE IONIZACIÓN ###
#######################################
E_0 = datos_P1["E_0"]*E_h
chi_H = -E_0

#############################
### LONGITUD ONDA TÉRMICA ###
#############################
lambda_e = np.sqrt(h**2/(2*np.pi*m_e*k*T_e))

##############################################
### POBLACIÓN HIDROGENOIDE (ECUACIÓN SAHA) ###
##############################################
n_H = 1/ ((Z-1)/n_e + 2*Z/(lambda_e*n_e**2)*1/G_H*np.exp(-chi_H/(k*T_e)))
E_l1_vals = datos_P1["E_l1"]
n_H_l1 = n_H/G_H*6*np.exp(-E_l1_vals*E_h/(k*T_e))

########################################
### POBLACIÓN NÚCLEOS (¿INNECESARIO? ###
########################################
n_p = (n_e-(Z-1)*n_H)/Z

#############################
### INTENSIDAD ESPECÍFICA ###
#############################
I = 0
for i in range(len(nu_if)):
    I += L*h/(4*np.pi)*nu_if[i]*n_H_l1[i]*A_if[i]*phi_V[i]

################
### GUARDADO ###
################
np.savez_compressed(
    'data/espectro_final.npz',
    nu = nu,
    I=I
)
