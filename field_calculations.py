""" MovingChargesField class calculates both the electric and magnetic fields 
generated from moving point charges, as well as the respective scalar and 
vector potentials (in the Lorenz gauge). These are determined numerically at each 
grid point by determining the retarded time of each point charge. The Liénard–Wiechert 
potentials and corresponding E and B field equations are then evaluated. 
"""
import numpy as np
import scipy.constants as constants
from scipy import optimize

# Constants
eps = constants.epsilon_0
mu = constants.mu_0
pi = constants.pi
e = constants.e
c = constants.c


class MovingChargesField():

    def __init__(self, charges, h=1e-20):
        """Determines the electric and magnetic fields (E and B) generated from 
        moving point charge(s) at the specified time and grid points.

        Args:
            charges (list of :obj: Charge)): Point charge object(s) that generate the fields
            h (float, optional): Tolerance for Newton's Method optimization. Defaults to 1e-20.
        """
        try:
            len(charges)
        except TypeError:
            charges = [charges]
        self.charges = charges
        self.h = h  # Causes overflow errors if too small

    def calculate_E(self, t, X, Y, Z, pcharge_field='Total', plane=False):
        """Calculates the electric field E generated from the point charge(s).

        Args:
            t (float): Time of simulation in seconds.
            X (:obj: ndarray(float, ndim=3)): meshgrid of X values in meters.
            Y (:obj: ndarray(float, ndim=3)): meshgrid of Y values in meters.
            Z (:obj: ndarray(float, ndim=3)): meshgrid of Z values in meters.
            pcharge_field (str, optional): Determines which field generated from the point
                charges is calculated: 'Velocity', 'Acceleration', or 'Total'. Defaults to 'Total'.
            plane(bool): True if meshgrid is 2 dimensional and returns 2D array. Defaults to False. 

        Returns:
            list of Ex, Ey, and Ez ndarrays which are 2 dimensional if plane is True, otherwise 3.

        """
        t_array = np.ones((X.shape))
        t_array[:, :, :] = t
        Ex = np.zeros((X.shape))
        Ey = np.zeros((X.shape))
        Ez = np.zeros((X.shape))
        for charge in self.charges:
            # Determine retarded time of charge to calculate E
            tr = optimize.newton(func=charge.retarded_time, x0=t_array,
                                 args=(t_array, X, Y, Z), tol=self.h)
            E_field = self._calculate_individual_E(
                charge, tr, X, Y, Z, pcharge_field)
            Ex += E_field[0]
            Ey += E_field[1]
            Ez += E_field[2]
        if plane:
            if X.shape[0] == 1:
                return(Ex[0, :, :], Ey[0, :, :], Ez[0, :, :])
            elif X.shape[1] == 1:
                return(Ex[:, 0, :], Ey[:, 0, :], Ez[:, 0, :])
            elif X.shape[2] == 1:
                return(Ex[:, :, 0], Ey[:, :, 0], Ez[:, :, 0])
        return (Ex, Ey, Ez)

    def calculate_B(self, t, X, Y, Z, pcharge_field='Total', plane=False):
        """Calculates the magnetic field B generated from the point charge(s).

        Args:
            t (float): Time of simulation in seconds.
            X (:obj: ndarray(float, ndim=3)): meshgrid of X values in meters.
            Y (:obj: ndarray(float, ndim=3)): meshgrid of Y values in meters.
            Z (:obj: ndarray(float, ndim=3)): meshgrid of Z values in meters.
            pcharge_field (str, optional): Determines which field generated from the point
                charges is calculated: 'Velocity', 'Acceleration', or 'Total'. Defaults to 'Total'.
            plane(bool): True if meshgrid is 2 dimensional and returns 2D array. Defaults to False. 

        Returns:
            list of Ex, Ey, and Ez ndarrays which are 2 dimensional if plane is True, otherwise 3.
        """
        t_array = np.ones((X.shape))
        t_array[:, :, :] = t
        Bx = np.zeros((X.shape))
        By = np.zeros((X.shape))
        Bz = np.zeros((X.shape))
        for charge in self.charges:
            # Determine retarded time of charge to calculate B
            tr = optimize.newton(func=charge.retarded_time, x0=t_array,
                                 args=(t_array, X, Y, Z), tol=self.h)
            Ex, Ey, Ez = self._calculate_individual_E(
                charge, tr, X, Y, Z, pcharge_field)
            rx = X - charge.xpos(tr)
            ry = Y - charge.ypos(tr)
            rz = Z - charge.zpos(tr)
            r_mag = (rx**2 + ry**2 + rz**2)**0.5
            # Griffiths Eq. 10.73
            Bx += 1/(c*r_mag)*(ry*Ez-rz*Ey)
            By += 1/(c*r_mag)*(rz*Ex-rx*Ez)
            Bz += 1/(c*r_mag)*(rx*Ey-ry*Ex)
        if plane:
            if X.shape[0] == 1:
                return(Bx[0, :, :], By[0, :, :], Bz[0, :, :])
            elif X.shape[1] == 1:
                return(Bx[:, 0, :], By[:, 0, :], Bz[:, 0, :])
            elif X.shape[2] == 1:
                return(Bx[:, :, 0], By[:, :, 0], Bz[:, :, 0])
        return (Bx, By, Bz)

    def _calculate_individual_E(self, charge, tr, X, Y, Z, pcharge_field):
        "Calculates the electric field generated from an individual point charge."
        # retarded position to field point - Griffiths Eq. 10.54
        rx = X - charge.xpos(tr)
        ry = Y - charge.ypos(tr)
        rz = Z - charge.zpos(tr)
        r_mag = (rx**2 + ry**2 + rz**2)**0.5
        vx = charge.xvel(tr)  # retarded velocity - Griffiths Eq. 10.54
        vy = charge.yvel(tr)
        vz = charge.zvel(tr)
        ax = charge.xacc(tr)  # retarded acceleration
        ay = charge.yacc(tr)
        az = charge.zacc(tr)
        ux = c*rx/r_mag - vx  # Griffiths Eq. 10.71
        uy = c*ry/r_mag - vy
        uz = c*rz/r_mag - vz
        r_dot_u = rx*ux + ry*uy + rz*uz
        r_dot_a = rx*ax + ry*ay + rz*az
        vel_mag = (vx**2 + vy**2 + vz**2)**0.5
        # Griffiths Eq. 10.72
        const = e/(4*pi*eps) * r_mag/(r_dot_u)**3
        if not charge.pos_charge:  # negative charge
            const *= -1
        xvel_field = const*(c**2-vel_mag**2)*ux
        yvel_field = const*(c**2-vel_mag**2)*uy
        zvel_field = const*(c**2-vel_mag**2)*uz
        # Using triple product rule to simplify Eq. 10.72
        xacc_field = const*(r_dot_a*ux - r_dot_u*ax)
        yacc_field = const*(r_dot_a*uy - r_dot_u*ay)
        zacc_field = const*(r_dot_a*uz - r_dot_u*az)
        if pcharge_field == 'Velocity':
            return (xvel_field, yvel_field, zvel_field)
        if pcharge_field == 'Acceleration':
            return (xacc_field, yacc_field, zacc_field)
        if pcharge_field == 'Total':
            return (xvel_field+xacc_field, yvel_field+yacc_field,
                    zvel_field+zacc_field)

    def calculate_potentials(self, t, X, Y, Z, plane=False):
        """Calculates the magnetic field B generated from the point charge(s).

        Args:
            t (float): Time of simulation in seconds.
            X (:obj: ndarray(float, ndim=3)): meshgrid of X values in meters.
            Y (:obj: ndarray(float, ndim=3)): meshgrid of Y values in meters.
            Z (:obj: ndarray(float, ndim=3)): meshgrid of Z values in meters.
            plane(bool): True if meshgrid is 2 dimensional and returns 2D array. Defaults to False. 

        Returns:
            list of V, Ax, Ay, and Az ndarrays which are 2 dimensional if plane is True, otherwise 3.
        """
        t_array = np.ones((X.shape))
        t_array[:, :, :] = t
        V = np.zeros((X.shape))
        Ax = np.zeros((X.shape))
        Ay = np.zeros((X.shape))
        Az = np.zeros((X.shape))
        for charge in self.charges:
            # Determine retarded time of charge to calculate potentials
            tr = optimize.newton(func=charge.retarded_time, x0=t_array,
                                 args=(t_array, X, Y, Z), tol=self.h)
            # retarded position to field point - Griffiths Eq. 10.54
            rx = X - charge.xpos(tr)
            ry = Y - charge.ypos(tr)
            rz = Z - charge.zpos(tr)
            r_mag = (rx**2 + ry**2 + rz**2)**0.5
            vx = charge.xvel(tr)  # retarded velocity - Griffiths Eq. 10.54
            vy = charge.yvel(tr)
            vz = charge.zvel(tr)
            r_dot_v = rx*vx + ry*vy + rz*vz
            # Griffiths Eq. 10.53
            if charge.pos_charge:
                individual_V = e*c/(4*pi*eps*(r_mag*c-r_dot_v))
            else:
                individual_V = -e*c/(4*pi*eps*(r_mag*c-r_dot_v))
            V += individual_V
            # Griffiths Eq. 10.53
            Ax += vx/c**2*individual_V
            Ay += vy/c**2*individual_V
            Az += vz/c**2*individual_V
        if plane:
            if X.shape[0] == 1:
                return(V[0, :, :], Ax[0, :, :], Ay[0, :, :], Az[0, :, :])
            elif X.shape[1] == 1:
                return(V[:, 0, :], Ax[:, 0, :], Ay[:, 0, :], Az[:, 0, :])
            elif X.shape[2] == 1:
                return(V[:, :, 0], Ax[:, :, 0], Ay[:, :, 0], Az[:, :, 0])
        return (V, Ax, Ay, Az)

    def calculate_Poynting(self, t, X, Y, Z, plane=False):
        """Calculates the Poynting vector S generated from the point charge(s).

        Args:
            t (float): Time of simulation in seconds.
            X (:obj: ndarray(float, ndim=3)): meshgrid of X values in meters.
            Y (:obj: ndarray(float, ndim=3)): meshgrid of Y values in meters.
            Z (:obj: ndarray(float, ndim=3)): meshgrid of Z values in meters.
            plane(bool): True if meshgrid is 2 dimensional and returns 2D array. Defaults to False. 

        Returns:
            S ndarray which are 2 dimensional if plane is True, otherwise 3.
        """
        t_array = np.ones((X.shape))
        t_array[:, :, :] = t
        Ex = np.zeros((X.shape))
        Ey = np.zeros((X.shape))
        Ez = np.zeros((X.shape))
        for charge in self.charges:
            # Determine retarded time of charge to calculate E
            tr = optimize.newton(func=charge.retarded_time, x0=t_array,
                                 args=(t_array, X, Y, Z), tol=self.h)
            E_field = self._calculate_individual_E(
                charge, tr, X, Y, Z, 'Acceleration')
            Ex += E_field[0]
            Ey += E_field[1]
            Ez += E_field[2]
        S = 1/(mu*c)*(Ex**2+Ey**2+Ez**2)  # Griffiths 11.67

        if plane:
            if X.shape[0] == 1:
                return S[0, :, :]
            elif X.shape[1] == 1:
                return S[:, 0, :]
            elif X.shape[2] == 1:
                return S[:, :, 0]
        return S
