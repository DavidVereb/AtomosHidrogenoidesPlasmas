import numpy as np
import scipy as sp

import math
from scipy.constants import physical_constants, k, epsilon_0, e, h, m_e, m_p, c, hbar, Rydberg

E_h, _, _ = physical_constants['Hartree energy']
a_0, _, _ = physical_constants['Bohr radius']

class Plasma:
    def __init__(self, atomic_number=1, plasma_length=1):
        """
        Instancia un objeto Plasma con métodos para el cálculo de resultados de interés.
        :param atomic_number: Número atómico del plasma.
        :param plasma_length: Espesor del plasma. Nuestro modelo solo es válido para plasmas delgados.
        """
        # Parámetros del plasma (SI)
        self.T_e = None
        self.n_e = None
        self.Z = atomic_number
        self.L = plasma_length

        # Discretización atómica
        self.N_r = 200_000
        self.max_radius = 500
        self.r = np.linspace(0, self.max_radius, self.N_r + 1)
        self.dr = self.r[1] - self.r[0]

        # Autoenergías y autofunciones
        self.n_max = 10 # Nos restringimos a n=2,...,10
        self.l_max = 3 # Nos restringimos a l=0,...3
        self.E_0 = None
        self.u_0 = None
        self.E_l = {}
        self.u_l = {}

        # Transiciones y coeficientes de Einstein
        self.nu_if = None
        self.A_if = None

        # Malla de frecuencias
        self.N_nu = int(1e5)
        self.nu_min = None
        self.nu_max = None
        self.nu = None

        # Poblaciones
        self.lambda_e = None
        self.G_H = None
        self.n_H = None
        self.n_P = None

        # Espectro
        self.I = None

    def set_temperature(self, temperature, units='K'):
        """
        Define la temperatura del plasma en Kelvin o electronvoltios.
        :param temperature: Temperatura electrónica del plasma.
        :param units: Si units == 'K', entonces las unidades de la temperatura son Kelvin. Si units == 'eV', las
        unidades de la temperatura son electronvoltios.
        """
        if units == 'K':
            self.T_e = temperature
        elif units == 'eV':
            self.T_e = e/k*temperature
        else:
            raise ValueError(f'Unidad de temperatura no valida: \'{units}\'. Las unidades permitidos son \'K\': Kelvin y \'eV\': electronvoltios.')

    def set_electron_density(self, electron_density, units='m'):
        """
        Define la densidad electrónica del plasma en m^-3 o en c^-3.
        :param electron_density: Densidad electrónica del plasma.
        :param units: Si units == 'm', entonces las unidades de la temperatura son m^-3. Si units == 'cm', las
        unidades de la temperatura son cm^-3.
        """
        if units == 'm':
            self.n_e = electron_density
        elif units == 'cm':
            self.n_e = 1e6*electron_density
        else:
            raise ValueError(f'Unidad de densidad electrónica no valida: \'{units}\'. Las unidades permitidos son \'m\': metros^-3 y \'cm\': centímetros^-3.')

    def set_atomic_mesh(self, num_intervals):
        """

        :param num_intervals: Número de intervalos de la discretización radial atómica.
        :return:
        """
        self.N_r = int(num_intervals)
        self.r = np.linspace(0, self.max_radius, self.N_r + 1)
        self.dr = self.r[1] - self.r[0]

    def analitical_schrodinger(self):
        pass

    def solve_schrodinger(self, plasma_effects=True):
        """
        Resuelve la ecuación de Schrödinger independiente del tiempo (ecuación de autovalores).
        :param plasma_effects: Si este parámetro es verdadero, se considera el potencial con efectos de plasma. En
        caso contrario, no.
        """
        # Constantes con dimensión
        r_D_si = np.sqrt(epsilon_0 * k * self.T_e / (self.n_e * e**2))  # Longitud de Debye
        r_0_si = np.cbrt(3 / (4 * np.pi * self.n_e))  # Radio de la esfera de neutralidad

        # Constantes adimensionales
        r_D = r_D_si / a_0  # Longitud de Debye adimensional
        kT_e = k * self.T_e / E_h  # ¿Energía térmica electrónica? Adimensional
        r_0 = r_0_si / a_0  # Radio de la esfera de neutralidad adimensional
        Gamma = self.Z**2 / (r_0 * kT_e)  # Parámetro de acoplamiento adimensional

        # Potencial adimensional
        if plasma_effects:
            #V = lambda _r: -self.Z / _r * np.exp(-_r / r_D) * (1 - Gamma + 0.5 * Gamma * _r / r_D)
            factor = Gamma / (Gamma + 1)
            V = lambda _r: -self.Z / _r * np.exp(-_r / r_D) * (1 - factor + factor * (1 - _r / r_D) ** 2)
        else:
            V = lambda _r: -self.Z / _r

        # Hamiltoniano
        r_inner = self.r[1:-1]
        first_diagonal = lambda _l: 1 / self.dr**2 + V(r_inner) + _l * (_l + 1) / (2 * r_inner**2)
        second_diagonal = -1 / (2 * self.dr**2) * np.ones(self.N_r - 2)

        # Cálculo del espectro
        lower_bound = -0.5 * self.Z**2 - 0.5
        E_dict = {}
        u_dict = {}
        for l in range(self.l_max+1):
            E, u_inner = sp.linalg.eigh_tridiagonal(first_diagonal(l), second_diagonal, select='v',
                                             select_range=(lower_bound, -1e-10))
            u = np.zeros((self.N_r + 1, len(E)))
            u[1:-1, :] = u_inner
            E_dict[l] = E
            u_dict[l] = u

            # Filtrado
            num_states_i = min(self.n_max - l, len(E_dict[l]))
            E_dict[l] = E_dict[l][:num_states_i]
            u_dict[l] = u_dict[l][:, :num_states_i]

            # Normalización
            norm = np.sqrt(np.trapezoid(np.abs(u_dict[l])**2, x=self.r, axis=0))
            u_dict[l] /= norm

        # Definimos autoenergías y autofunciones del plasma en Hartrees
        if len(E_dict[0]) > 0:
            self.E_0 = E_h * E_dict[0][0]
            self.u_0 = u_dict[0][:, 0]

            self.E_l[0] = E_h * E_dict[0][1:]
            self.u_l[0] = u_dict[0][:, 1:]
            for l in range(1, self.l_max + 1):
                self.E_l[l] = E_h * E_dict[l]
                self.u_l[l] = u_dict[l]
        else:
            print("¡Aviso! El plasma es tan denso que no existen estados ligados.")
            self.E_0 = None
            self.u_0 = None

            self.E_l[0] = []
            self.u_l[0] = []
            for l in range(1, self.l_max + 1):
                self.E_l[l] = []
                self.u_l[l] = []

    def solve_transitions(self):
        # Elementos matriz transiciones (angular)
        x, w = np.polynomial.legendre.leggauss(20)

        # Estamos estudiando la serie Lyman (n_f = 1)
        l_f = 0
        m_f = 0

        # Regla de selección
        l_i = 1

        N_lm = lambda l, m: (-1)**m * np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))

        factor_angular = 0

        for q in [-1, 0, 1]:
            # Regla de selección
            m_i = -q

            # Polinomios de Legendre
            P_final = sp.special.lpmv(m_f, l_f, x)
            P_foton = sp.special.lpmv(q, 1, x)
            P_inicial = sp.special.lpmv(m_i, l_i, x)

            # Armónicos esféricos
            Y_final = N_lm(l_f, m_f) * P_final
            Y_foton = N_lm(1, q) * P_foton
            Y_inicial = N_lm(l_i, m_i) * P_inicial

            integral_theta = np.sum(w * Y_final * Y_foton * Y_inicial)
            integral_phi = 2*np.pi
            factor_angular += (integral_theta * integral_phi)**2

        # Elementos de matriz al cuadrado: S_if
        E_f = self.E_0
        u_f = self.u_0
        E_i = self.E_l[1]
        u_i = self.u_l[1]

        S_if = np.zeros(len(E_i))
        for n in range(len(E_i)):
            integral_radial = np.trapezoid(np.conj(u_f) * self.r * u_i[:, n], x = self.r)
            S_if[n] = 4 * np.pi / 3 * integral_radial**2 * factor_angular

        # Usamos Sistema Internacional a partir de ahora.
        self.nu_if = (E_i - E_f) / h
        self.A_if = 8 / 3 * np.pi**2 * (a_0 * e)**2 / (epsilon_0 * hbar * c**3) * self.nu_if**3 * S_if

    def solve_population(self, plasma_effects=True):
        # Función de partición
        self.G_H = 2 # Incluimos el estado fundamental n=1, l=0
        if plasma_effects:
            for l in range(self.l_max+1):
                g_l = 2 * (2 * l + 1)
                for E_nl in self.E_l[l]:
                    self.G_H += g_l * np.exp(-(E_nl - self.E_0 ) / (k * self.T_e))
        else:
            for n in range(2, len(self.E_l[0])+2):
                g_n = 2 * n**2
                E_n = self.E_l[0][n-2]
                self.G_H += g_n * np.exp(-(E_n - self.E_0) / (k * self.T_e))

        # Energía potencial de ionización
        chi_H = -self.E_0

        # Longitud de onda térmica
        self.lambda_e = np.sqrt(h**2 / (2 * np.pi * m_e * k * self.T_e))

        # Población neutra (ecuación de Saha)
        C = 2 / (self.lambda_e**3 * self.n_e * self.G_H)* np.exp(-chi_H / (k * self.T_e))
        self.n_H = self.n_e / ((C + 1) * self.Z - 1)

        # Población núcleos
        self.n_P = (self.n_e - (self.Z - 1) * self.n_H) / self.Z


    def solve_spectrum(self):
        # Parámetros perfil de linea
        sigma = np.sqrt(k * self.T_e / (m_p * c ** 2)) * self.nu_if
        n_i = np.arange(2, 2 + len(self.nu_if))
        """
        Original del guion
        gamma_L = 8 * np.pi**2 * self.n_e / (6 * np.sqrt(3)) * (hbar / m_e)**2 * np.sqrt(2 * m_e /
                (np.pi * k * self.T_e)) * (0.9 - 1.1 / self.Z) * (3 * n_i / (2 * self.Z))**2 * (n_i**2 - 3)
        # ¿Guion puede estar mal?
        gamma_L = np.abs(gamma_L)
        """
        gamma_L = 8 * np.pi**2 * self.n_e / (6 * np.sqrt(3)) * (hbar / m_e)**2 * np.sqrt(2 * m_e /
                (np.pi * k * self.T_e)) * (3 * n_i / (2 * self.Z))**2 * (n_i**2 - 3)

        # Malla frecuencias
        self.nu_min = self.nu_if[0] * 0.95
        self.nu_max = self.nu_if[-1] * 1.05
        self.nu = np.linspace(self.nu_min, self.nu_max, self.N_nu)

        # Perfil de línea
        phi_V = []
        for n in range(len(self.nu_if)):
            perfil = sp.special.voigt_profile(self.nu - self.nu_if[n], sigma[n], gamma_L[n]/2)
            phi_V.append(perfil)

        # Intensidad específica
        n_H_l1 = 6 * self.n_H / self.G_H * np.exp(-(self.E_l[1]-self.E_0) / (k * self.T_e))

        self.I = 0
        for n in range(len(self.nu_if)):
            self.I += self.L * h / (4 * np.pi) * self.nu_if[n] * n_H_l1[n] * self.A_if[n] * phi_V[n]

    def solve_plasma(self, plasma_effects=True):
        print("Resolviendo ecuación de autovalores...")
        self.solve_schrodinger(plasma_effects)
        print("Resolviendo transiciones...")
        self.solve_transitions()
        print("Resolviendo poblaciones...")
        self.solve_population(plasma_effects)
        print("Resolviendo espectro...")
        self.solve_spectrum()