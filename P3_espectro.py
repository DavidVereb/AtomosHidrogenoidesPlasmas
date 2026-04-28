"""
Carga "data/coeficientes_einstein.npz" y los parámetros de config.py. Resuelve el sistema de Saha-Bolzmann para sacar
las poblaciones n_Hnl. Construye el perfil de Voigt para cada línea espectral combinando los ensanchamientos Doppler y
colisional. Finalmente suma todo para calcular la emisividad j(v) y la intensidad específica I(v).

Exporta en "data/espectro_final.npz" el vector de frecuencias v y el vector de intensidad específica I(v) de la serie
Lyman.
"""

