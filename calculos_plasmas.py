import numpy as np
import scipy.linalg as la

from scipy.constants import physical_constants, k, epsilon_0, e, h, m_e, m_p, c, hbar

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
        self.max_radius = 350
        self.N = 200_000
        self.r = np.linspace(0, self.max_radius, self.N + 1)
        self.dr = self.r[1] - self.r[0]

        # Autoenergías y autofunciones
        self.E_0 = None
        self.u_0 = None
        self.E_l = {}
        self.u_l = {}

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
        self.N = int(num_intervals)
        self.r = np.linspace(0, self.max_radius, self.N + 1)
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
        r_D_si = np.sqrt(epsilon_0 * k * self.T_e / (self.n_e * e ** 2))  # Longitud de Debye
        r_0_si = np.cbrt(3 / (4 * np.pi * self.n_e))  # Radio de la esfera de neutralidad

        # Constantes adimensionales
        r_D = r_D_si / a_0  # Longitud de Debye adimensional
        kT_e = k * self.T_e / E_h  # ¿Energía térmica electrónica? Adimensional
        r_0 = r_0_si / a_0  # Radio de la esfera de neutralidad adimensional
        Gamma = self.Z ** 2 / (r_0 * kT_e)  # Parámetro de acoplamiento adimensional

        # Potencial adimensional
        if plasma_effects:
            V = lambda _r: -self.Z / _r * np.exp(-_r / r_D) * (1 - Gamma + 1 / 2 * Gamma * _r / r_D)
        else:
            V = lambda _r: -self.Z / _r

        # Hamiltoniano
        r_inner = self.r[1:-1]
        first_diagonal = lambda _l: 1 / self.dr ** 2 + V(r_inner) + _l * (_l + 1) / (2 * r_inner ** 2)
        second_diagonal = -1 / (2 * self.dr ** 2) * np.ones(self.N - 2)

        # Cálculo del espectro
        lower_bound = -0.5 * self.Z ** 2 - 0.5
        E_dict = {}
        u_dict = {}
        n_max = 10  # Nos restringimos a n=2,...,10
        l_max = 3 # Nos restringimos a l=0,...3
        for l in range(l_max+1):
            E, u_inner = la.eigh_tridiagonal(first_diagonal(l), second_diagonal, select='v',
                                             select_range=(lower_bound, -1e-6))
            u = np.zeros((self.N + 1, len(E)))
            u[1:-1, :] = u_inner
            E_dict[l] = E
            u_dict[l] = u

            # Filtrado
            num_states_i = min(n_max - l, len(E_dict[l]))
            E_dict[l] = E_dict[l][:num_states_i]
            u_dict[l] = u_dict[l][:, :num_states_i]

            # Normalización
            norm = np.sqrt(np.trapezoid(np.abs(u_dict[l]) ** 2, x=self.r, axis=0))
            u_dict[l] /= norm

        # Definimos autoenergías y autofunciones del plasma
        self.E_0 = E_h*E_dict[0][0]
        self.u_0 = E_h*u_dict[0][:, 0]

        self.E_l[0] = E_h * E_dict[0][1:]
        self.u_l[0] = E_h * u_dict[0][:, 1:]
        for l in range(1, l_max+1):
            self.E_l[l] = E_h * E_dict[l]
            self.u_l[l] = E_h * u_dict[l]