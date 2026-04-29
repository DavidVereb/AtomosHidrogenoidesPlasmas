"""
Script de comprobación visual para las funciones de onda del átomo en plasma.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .config import *

def plot_frontera(l, n):
    datos = np.load('data/funciones_onda.npz')

    u_matriz = datos[f'u_l{l}']

    i = n - l - 1

    if i < 0 or i >= u_matriz.shape[1]:
        print(f"Error: El estado n={n} no existe físicamente para l={l} con los parámetros actuales.")
        return

    u_target = u_matriz[:, i]

    plt.figure(figsize=(8, 4))
    plt.plot(r, u_target, label=f'Densidad $|u(r)|^2$ para $n={n}$')
    plt.axvline(100, color='red', linestyle='--', label='Radio clásico (100 u.a.)')

    plt.title(f'Comprobación de la frontera para $l={l}, n={n}$')
    plt.xlabel(r'$r$ (unidades atómicas)')
    plt.ylabel('Densidad de probabilidad')
    plt.xlim(0, R_max)
    plt.grid(True)
    plt.legend()

    nombre_archivo = f'comprobacion_l{l}_n{n}.png'
    plt.savefig(nombre_archivo, dpi=300)
    print(f"Gráfica guardada exitosamente como '{nombre_archivo}'")

if __name__ == '__main__':
    plot_frontera(0, 10)