import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit


class StellarDynamics(object):
    """ Stellar dynamics class for binary stars. """

    def __init__(self):
        # System params.
        self.period = None  # [days].
        self.time_of_periastron = None  # [day].
        self.eccentricity = None  # [ ].
        self.argument_of_periastron_primary = None  # [radians].

        # Spectroscopic params.
        self.semi_amplitude_primary = None  # [km/s].
        self.semi_amplitude_secondary = None  # [km/s].
        self.rv_offset_primary = None  # [km/s].
        self.rv_offset_secondary = None  # [km/s].

        # Astrometric params.
        self.semi_major_axis = None  # [au].
        self.inclination = None  # [radians].
        self.longitude_of_ascending_node = None  # [radians].

        # Fixed params.
        self.parallax = None  # [arcseconds].

        # Computed params.
        self.jd = None  # [days].
        self.phase = None  # [ ].
        self.mean_anomaly = None  # [radians].
        self.eccentric_anomaly = None  # [radians].
        self.true_anomaly = None  # [radians].

        # Derived params.
        self.semi_major_axis_primary = None  # [au].
        self.semi_major_axis_secondary = None  # [au].
        self.total_mass = None  # [m_solar].
        self.q_primary = None  # [ ].
        self.q_secondary = None  # [ ].
        self.minimum_mass_primary = None  # [m_solar].
        self.minimum_mass_secondary = None  # [m_solar].
        self.mass_primary = None  # [m_solar].
        self.mass_secondary = None  # [m_solar].

    def __repr__(self):
        return 'Binary dynamics.'

    def auto_calculate_params(self, phase=False, semi_major_axis_via_ks=False,
                              semi_major_axis_via_masses=False,
                              semi_amplitude_via_masses=False):
        """ Auto calculate params. """
        if phase:
            # Always used when jd given.
            self.phase = ((self.jd - self.time_of_periastron)
                          % self.period) / self.period

        if semi_major_axis_via_ks:
            # Used for SB2 + astrometry cases.
            # Input {P, T0, e, w, k1, k2, g1, g2, i, Omega}.
            self._compute_mass_ratio()
            self._compute_minimum_mass()
            self._compute_individual_masses()

            self.semi_major_axis = \
                (((self.period * (24 * 3600)) ** 2)
                 * ((self.mass_primary + self.mass_secondary) * 1.98847e30)
                 * gravitational_constant / (4 * np.pi**2)) ** (1 / 3) \
                / astronomical_unit

        if semi_major_axis_via_masses:
            # Used for SB2 + astrometry cases.
            # Input {P, T0, e, w, g1, g2, i, Omega, m1, m2}.
            self.semi_major_axis = \
                (((self.period * (24 * 3600)) ** 2)
                 * ((self.mass_primary + self.mass_secondary) * 1.98847e30)
                 * gravitational_constant / (4 * np.pi**2)) ** (1 / 3) \
                / astronomical_unit

        if semi_amplitude_via_masses:
            # Used for SB2 + astrometry cases.
            # Input {P, T0, e, w, g1, g2, i, Omega, m1, m2}.
            self.semi_amplitude_primary = \
                (4 * np.pi**2 * gravitational_constant**2)**(1/6) \
                * (1 / np.sqrt(1 - (self.eccentricity ** 2))) \
                * (self.mass_secondary * 1.98847e30) \
                * np.sin(self.inclination) \
                * (((self.mass_primary + self.mass_secondary) * 1.98847e30) ** (-2 / 3)) \
                * ((self.period * (24 * 3600)) ** (-1 / 3))\
                / 1e3
            self.semi_amplitude_secondary = \
                (4 * np.pi**2 * gravitational_constant**2)**(1/6) \
                * (1 / np.sqrt(1 - (self.eccentricity ** 2))) \
                * (self.mass_primary * 1.98847e30) \
                * np.sin(self.inclination) \
                * (((self.mass_primary + self.mass_secondary) * 1.98847e30) ** (-2 / 3)) \
                * ((self.period * (24 * 3600)) ** (-1 / 3))\
                / 1e3

    def auto_derive_quantities(self, sb1=False, sb2=False, astrometry=False):
        """ Auto derive quantities. """
        if sb1 and not astrometry:
            # None to derive.
            pass

        if sb2 and not astrometry:
            # Can derive mass ratio and minimum mass.
            self._compute_mass_ratio()
            self._compute_minimum_mass()

        if astrometry:
            # Can derive total mass.
            self._compute_total_mass()

        if sb1 and astrometry:
            # Can derive individual masses.
            self._compute_total_mass()
            self._compute_masses_via_k()

        if sb2 and astrometry:
            # Can derive individual masses.
            self._compute_mass_ratio()
            self._compute_minimum_mass()
            self._compute_total_mass()
            self._compute_individual_masses()

    def _compute_mass_ratio(self):
        """ Compute mass ratio. """
        # Assert correct params have been set.
        try:
            assert self.semi_amplitude_primary is not None
            assert self.semi_amplitude_secondary is not None
        except AssertionError as err:
            raise AssertionError(
                'Mass ratio calculation requires params semi_amplitude_primary, '
                'semi_amplitude_secondary to be set.')

        self.q_primary = self.semi_amplitude_primary / self.semi_amplitude_secondary
        self.q_secondary = self.semi_amplitude_secondary / self.semi_amplitude_primary

    def _compute_minimum_mass(self):
        """ Compute minimum mass. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.eccentricity is not None
            assert self.q_primary is not None
            assert self.q_secondary is not None
        except AssertionError as err:
            raise AssertionError(
                'Minimum mass calculation requires params period, eccentricity'
                ', q_primary, q_secondary to be set.')

        self.minimum_mass_primary = \
            ((self.semi_amplitude_primary * 1e3)**3 * self.period * (24 * 3600)
             * (1 - self.eccentricity**2)**(3/2) * (1 + self.q_primary)**2) / \
            (2 * np.pi * gravitational_constant * self.q_primary**3) \
            / 1.98847e30
        self.minimum_mass_secondary = \
            ((self.semi_amplitude_secondary * 1e3)**3 * self.period * (24 * 3600)
             * (1 - self.eccentricity**2)**(3/2) * (1 + self.q_secondary)**2) / \
            (2 * np.pi * gravitational_constant * self.q_secondary**3) \
            / 1.98847e30

    def _compute_total_mass(self):
        """ Compute total mass. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.semi_major_axis is not None
        except AssertionError as err:
            raise AssertionError(
                'Total mass calculation requires params period, '
                'semi_major_axis to be set.')

        self.total_mass = 4 * np.pi ** 2 * \
                          (self.semi_major_axis * astronomical_unit) ** 3 \
                          / (gravitational_constant * (self.period * (24 * 3600)) ** 2) \
                          / 1.98847e30

    def _compute_masses_via_k(self):
        """ Compute individual mass via k. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.eccentricity is not None
            assert self.inclination is not None
            assert self.semi_amplitude_primary is not None
            assert self.total_mass is not None
        except AssertionError as err:
            raise AssertionError(
                'Individual mass via k calculation requires params period, '
                'eccentricity, inclination, semi_amplitude_primary, total_m'
                'ass to be set.')

        self.mass_secondary = self.semi_amplitude_primary * 1e3 \
                              * (self.total_mass * 1.98847e30)**(2/3) \
                              * (self.period * (24 * 3600))**(1/3) \
                              * (1 - self.eccentricity**2)**(1/2) \
                              / (4 * np.pi**2 * gravitational_constant**2)**(1/6) \
                              / np.sin(self.inclination) \
                              / 1.98847e30
        self.mass_primary = self.total_mass - self.mass_secondary

    def _compute_individual_masses(self):
        """ Compute individual mass. """
        # Assert correct params have been set.
        try:
            assert self.inclination is not None
            assert self.minimum_mass_primary is not None
            assert self.minimum_mass_secondary is not None
        except AssertionError as err:
            raise AssertionError(
                'Individual mass calculation requires params inclination, '
                'minimum_mass_primary, minimum_mass_secondary to be set.')

        self.mass_primary = self.minimum_mass_primary / np.sin(self.inclination) ** 3
        self.mass_secondary = self.minimum_mass_secondary / np.sin(self.inclination) ** 3

    def _compute_semi_major_axes(self):
        """ Compute semi-major axes. """
        # Assert correct params have been set.
        try:
            assert self.semi_major_axis is not None
            assert self.mass_primary is not None
            assert self.mass_secondary is not None
        except AssertionError as err:
            raise AssertionError(
                'Semi-major axes calculation requires params semi_major_axis, '
                'mass_primary, mass_secondary to be set.')

        self.semi_major_axis_primary = \
            self.semi_major_axis * (self.mass_secondary
                                    / (self.mass_primary + self.mass_secondary))
        self.semi_major_axis_secondary = \
            self.semi_major_axis * (self.mass_primary
                                    / (self.mass_primary + self.mass_secondary))

    @property
    def keplerian_radial_velocity_primary(self):
        """ Calculate primary Keplerian radial velocities. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.time_of_periastron is not None
            assert self.eccentricity is not None
            assert self.argument_of_periastron_primary is not None
            assert self.semi_amplitude_primary is not None
            assert self.rv_offset_primary is not None
        except AssertionError as err:
            raise AssertionError(
                'Keplerian model for radial velocity requires params period, '
                'time_of_periastron, eccentricity, argument_of_periastron_pri'
                'mary, semi_amplitude_primary, rv_offset_primary to be set.')

        # Calc orbital anomalies: mean, eccentric, true.
        self._orbital_anomalies()

        # LOS projection of Keplerian velocity.
        rvs = self.semi_amplitude_primary * (
            (np.cos(self.argument_of_periastron_primary + self.true_anomaly))
            + (self.eccentricity * np.cos(self.argument_of_periastron_primary)))

        # Systemic velocity constant.
        rvs += self.rv_offset_primary

        return rvs

    @property
    def keplerian_radial_velocity_secondary(self):
        """ Calculate secondary Keplerian radial velocities. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.time_of_periastron is not None
            assert self.eccentricity is not None
            assert self.argument_of_periastron_primary is not None
            assert self.semi_amplitude_secondary is not None
            assert self.rv_offset_secondary is not None
        except AssertionError as err:
            raise AssertionError(
                'Keplerian model for radial velocity requires params period, '
                'time_of_periastron, eccentricity, argument_of_periastron_pri'
                'mary, semi_amplitude_primary, rv_offset_primary to be set.')

        # Calc orbital anomalies: mean, eccentric, true.
        self._orbital_anomalies()

        # LOS projection of Keplerian velocity.
        rvs = self.semi_amplitude_secondary * (
            (np.cos(self.argument_of_periastron_primary + np.pi + self.true_anomaly))
            + (self.eccentricity * np.cos(self.argument_of_periastron_primary + np.pi)))

        # Systemic velocity constant.
        rvs += self.rv_offset_secondary

        return rvs

    @property
    def relative_astrometric_position_primary(self):
        """ Calculate relative astrometric positions: secondary at origin. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.time_of_periastron is not None
            assert self.eccentricity is not None
            assert self.argument_of_periastron_primary is not None
            assert self.semi_major_axis is not None
            assert self.inclination is not None
            assert self.longitude_of_ascending_node is not None
        except AssertionError as err:
            raise AssertionError(
                'Keplerian model for relative astrometric positions requires '
                'params period, time_of_periastron, eccentricity, argument_of'
                '_periastron_primary, semi_major_axis, inclination, longitude'
                '_of_ascending_node to be set.')

        # Calc orbital anomalies: mean, eccentric, true.
        self._orbital_anomalies()

        # Calc r(theta). NB. +a to put primary at origin and
        # draw secondary motion relative to it.
        r = self.semi_major_axis * (1.0 - self.eccentricity ** 2) / \
            (1 + self.eccentricity * np.cos(self.true_anomaly))
        if self.parallax is not None:
            # convert r [au] into r [arcseconds].
            r *= self.parallax

        # Cartesian co-ordinates in orbital plane.
        x = r * np.cos(self.true_anomaly)
        y = r * np.sin(self.true_anomaly)

        # Cartesian co-ordinates in sky plane.
        X, Y, Z = self._transform_orbit_to_sky_plane(x, y)

        # Calc measured quantities.
        rho = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)

        return X, Y, Z, rho, theta

    @property
    def relative_astrometric_position_secondary(self):
        """ Calculate relative astrometric positions: primary at origin. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.time_of_periastron is not None
            assert self.eccentricity is not None
            assert self.argument_of_periastron_primary is not None
            assert self.semi_major_axis is not None
            assert self.inclination is not None
            assert self.longitude_of_ascending_node is not None
        except AssertionError as err:
            raise AssertionError(
                'Keplerian model for relative astrometric positions requires '
                'params period, time_of_periastron, eccentricity, argument_of'
                '_periastron_primary, semi_major_axis, inclination, longitude'
                '_of_ascending_node to be set.')

        # Calc orbital anomalies: mean, eccentric, true.
        self._orbital_anomalies()

        # Calc r(theta). NB. -a to put primary at origin and
        # draw secondary motion relative to it.
        r = -self.semi_major_axis * (1.0 - self.eccentricity ** 2) / \
            (1 + self.eccentricity * np.cos(self.true_anomaly))
        if self.parallax is not None:
            # convert r [au] into r [arcseconds].
            r *= self.parallax

        # Cartesian co-ordinates in orbital plane.
        x = r * np.cos(self.true_anomaly)
        y = r * np.sin(self.true_anomaly)

        # Cartesian co-ordinates in sky plane.
        X, Y, Z = self._transform_orbit_to_sky_plane(x, y)

        # Calc measured quantities.
        rho = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)

        return X, Y, Z, rho, theta

    def _orbital_anomalies(self):
        """ Orbital anomalies: mean, eccentric, true. """
        # Mean anomaly.
        self.mean_anomaly = self.phase * 2 * np.pi

        # Eccentric anomaly.
        self.eccentric_anomaly = np.empty(self.mean_anomaly.shape)
        for i_ma, ma in enumerate(self.mean_anomaly):

            try:
                if self.eccentricity < 0.8:
                    ea = self._newtonian_raphson_method_for_keplers_equation(
                        E_i=ma, M=ma, e=self.eccentricity)
                else:
                    ea = self._newtonian_raphson_method_for_keplers_equation(
                        E_i=ma, M=ma, e=self.eccentricity)

                # Save solution.
                self.eccentric_anomaly[i_ma] = ea

            except RecursionError as err:
                print('Recursion error caught whilst doing '
                      'newton-raphson iteration of Keplers equation.')
                self.eccentric_anomaly[i_ma] = ma = np.nan

        # True anomaly.
        numerator = np.sqrt(1 + self.eccentricity) \
                * np.sin(self.eccentric_anomaly / 2)
        denominator = np.sqrt(1 - self.eccentricity) \
                * np.cos(self.eccentric_anomaly / 2)
        self.true_anomaly = 2 * np.arctan2(numerator, denominator)

    def _newtonian_raphson_method_for_keplers_equation(self, E_i, M, e):
        """ Numerical solution to Kepler's equation. """
        # Iteration step.
        E_i_1 = (M + (e * (np.sin(E_i) - (E_i * np.cos(E_i)))))\
                / (1 - (e * np.cos(E_i)))

        # Convergence test.
        epsilon = 1e-10
        if not abs(E_i_1 - E_i) < epsilon:
            return self._newtonian_raphson_method_for_keplers_equation(
                E_i_1, M, e)

        # Solution converged.
        return E_i_1

    def _transform_orbit_to_sky_plane(self, x, y):
        """ Transform co-ordinates from orbital to sky plane. """
        # Rotate about z by w.
        x1 = np.cos(self.argument_of_periastron_primary) * x \
             - np.sin(self.argument_of_periastron_primary) * y
        y1 = np.sin(self.argument_of_periastron_primary) * x \
             + np.cos(self.argument_of_periastron_primary) * y

        # Rotate about x1 by -i.
        x2 = x1
        y2 = np.cos(self.inclination) * y1

        # Rotate about Z by Omega.
        X = np.cos(self.longitude_of_ascending_node) * x2 \
            - np.sin(self.longitude_of_ascending_node) * y2
        Y = np.sin(self.longitude_of_ascending_node) * x2 \
            + np.cos(self.longitude_of_ascending_node) * y2
        Z = -np.sin(self.inclination) * y1

        return X, Y, Z
