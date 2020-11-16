"""Classes of charges used for the MovingChargesField class.
Charge class is abstract class for LinearAcceleratingCharge, 
LinearDeceleratingCharge, LinearVelocityCharge, OrbittingCharge, 
and OscillatingCharge classes.
"""
import numpy as np
from abc import ABC, abstractmethod
import scipy.constants as constants

# Constants
eps = constants.epsilon_0
pi = constants.pi
e = constants.e
c = constants.c
u_0 = constants.mu_0


class Charge(ABC):

    def __init__(self, pos_charge=True):
        self.pos_charge = pos_charge

    @abstractmethod
    def xpos(self, t):
        pass

    @abstractmethod
    def ypos(self, t):
        pass

    @abstractmethod
    def zpos(self, t):
        pass

    @abstractmethod
    def xvel(self, t):
        pass

    @abstractmethod
    def yvel(self, t):
        pass

    @abstractmethod
    def zvel(self, t):
        pass

    @abstractmethod
    def xacc(self, t):
        pass

    @abstractmethod
    def yacc(self, t):
        pass

    @abstractmethod
    def zacc(self, t):
        pass

    def retarded_time(self, tr, t, X, Y, Z):
        """Returns equation to solve for retarded time - Griffiths Eq. 10.55"""
        return ((X-self.xpos(tr))**2 + (Y-self.ypos(tr))**2 + (Z-self.zpos(tr))**2)**0.5 - c*(t-tr)


class OscillatingCharge(Charge):

    def __init__(self, pos_charge=True, start_position=(-2e-9, 0, 0),
                 direction=(1, 0, 0), amplitude=2e-9, max_speed=0.9*c, start_zero=False, stop_t=None):
        super().__init__(pos_charge)
        self.start_position = np.array(start_position)
        self.direction = np.array(direction) \
            / np.linalg.norm(np.array(direction))
        self.amplitude = amplitude
        self.w = max_speed/amplitude
        self.start_zero = start_zero
        self.stop_t = stop_t

    def xpos(self, t):
        xpos = self.start_position[0] \
            + self.direction[0]*self.amplitude*(1-np.cos(self.w*t))
        if self.start_zero:
            xpos[t < 0] = self.start_position[0]
        if self.stop_t is not None:
            xpos[t > self.stop_t] = self.start_position[0] \
                + self.direction[0]*self.amplitude * \
                (1-np.cos(self.w*self.stop_t))
        return xpos

    def ypos(self, t):
        ypos = self.start_position[1] \
            + self.direction[1]*self.amplitude*(1-np.cos(self.w*t))
        if self.start_zero:
            ypos[t < 0] = self.start_position[1]
        if self.stop_t is not None:
            ypos[t > self.stop_t] = self.start_position[1] \
                + self.direction[1]*self.amplitude * \
                (1-np.cos(self.w*self.stop_t))
        return ypos

    def zpos(self, t):
        zpos = self.start_position[2] \
            + self.direction[2]*self.amplitude*(1-np.cos(self.w*t))
        if self.start_zero:
            zpos[t < 0] = self.start_position[2]
        if self.stop_t is not None:
            zpos[t > self.stop_t] = self.start_position[2] \
                + self.direction[2]*self.amplitude * \
                (1-np.cos(self.w*self.stop_t))
        return zpos

    def xvel(self, t):
        xvel = self.direction[0]*self.amplitude*self.w*np.sin(self.w*t)
        if self.start_zero:
            xvel[t < 0] = 0
        if self.stop_t is not None:
            xvel[t > self.stop_t] = 0
        return xvel

    def yvel(self, t):
        yvel = self.direction[1]*self.amplitude*self.w*np.sin(self.w*t)
        if self.start_zero:
            yvel[t < 0] = 0
        if self.stop_t is not None:
            yvel[t > self.stop_t] = 0
        return yvel

    def zvel(self, t):
        zvel = self.direction[2]*self.amplitude*self.w*np.sin(self.w*t)
        if self.start_zero:
            zvel[t < 0] = 0
        if self.stop_t is not None:
            zvel[t > self.stop_t] = 0
        return zvel

    def xacc(self, t):
        xacc = self.direction[0]*self.amplitude*self.w**2*np.cos(self.w*t)
        if self.start_zero:
            xacc[t < 0] = 0
        if self.stop_t is not None:
            xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t):
        yacc = self.direction[1]*self.amplitude*self.w**2*np.cos(self.w*t)
        if self.start_zero:
            yacc[t < 0] = 0
        if self.stop_t is not None:
            yacc[t > self.stop_t] = 0
        return yacc

    def zacc(self, t):
        zacc = self.direction[2]*self.amplitude*self.w**2*np.cos(self.w*t)
        if self.start_zero:
            zacc[t < 0] = 0
        if self.stop_t is not None:
            zacc[t > self.stop_t] = 0
        return zacc

    def get_period(self):
        return 2*np.pi/self.w


class OrbittingCharge(Charge):

    def __init__(self, pos_charge=True, phase=0, amplitude=2e-9,
                 max_speed=0.9*c, start_zero=False):
        """Second charge start position is negative x."""
        super().__init__(pos_charge)
        self.amplitude = amplitude
        self.w = max_speed/self.amplitude
        self.start_zero = start_zero
        self.phase = phase

    def xpos(self, t):
        xpos = self.amplitude*np.cos(self.w*t+self.phase)
        if self.start_zero:
            xpos[t < 0] = self.amplitude*np.cos(self.phase)
        return xpos

    def ypos(self, t):
        ypos = self.amplitude*np.sin(self.w*t+self.phase)
        if self.start_zero:
            ypos[t < 0] = self.amplitude*np.sin(self.phase)
        return ypos

    def zpos(self, t):
        return 0

    def xvel(self, t):
        xvel = -self.amplitude*self.w*np.sin(self.w*t+self.phase)
        if self.start_zero:
            xvel[t < 0] = 0
        return xvel

    def yvel(self, t):
        yvel = self.amplitude*self.w*np.cos(self.w*t+self.phase)
        if self.start_zero:
            yvel[t < 0] = 0
        return yvel

    def zvel(self, t):
        return 0

    def xacc(self, t):
        xacc = -self.amplitude*self.w**2*np.cos(self.w*t+self.phase)
        if self.start_zero:
            xacc[t < 0] = 0
        return xacc

    def yacc(self, t):
        yacc = -self.amplitude*self.w**2*np.sin(self.w*t+self.phase)
        if self.start_zero:
            yacc[t < 0] = 0
        return yacc

    def zacc(self, t):
        return 0

    def get_period(self):
        return 2*np.pi/self.w


class OscillatingOrbittingCharge(Charge):

    def __init__(self, pos_charge=True, phase=0, radius=2e-9,
                 w=60*2*np.pi, max_speed=0.9*c, start_zero=False):
        """Second charge start position is negative x."""
        super().__init__(pos_charge)
        self.phase = phase
        self.radius = radius
        self.w = w
        self.max_speed = max_speed
        self.A = max_speed/(radius*w)
        self.start_zero = start_zero

    def xpos(self, t):
        xpos = self.radius*np.cos(self.A*np.cos(self.w*t)+self.phase)
        if self.start_zero:
            xpos[t < 0] = self.radius*np.cos(self.phase)
        return xpos

    def ypos(self, t):
        ypos = self.radius*np.sin(self.A*np.cos(self.w*t)+self.phase)
        if self.start_zero:
            ypos[t < 0] = self.radius*np.sin(self.phase)
        return ypos

    def zpos(self, t):
        return 0

    def xvel(self, t):
        xvel = self.max_speed * \
            (np.sin(t*self.w)*np.sin(self.A*np.cos(t*self.w)+self.phase))
        if self.start_zero:
            xvel[t < 0] = 0
        return xvel

    def yvel(self, t):
        yvel = -self.max_speed * \
            (np.sin(t*self.w)*np.cos(self.A*np.cos(t*self.w)+self.phase))
        if self.start_zero:
            yvel[t < 0] = 0
        return yvel

    def zvel(self, t):
        return 0

    def xacc(self, t):
        xacc = self.max_speed*self.w*(np.cos(t*self.w)*np.sin(self.A*np.cos(t*self.w)+self.phase)
                                      - self.A*np.sin(t*self.w)**2*np.cos(self.A*np.cos(t*self.w)+self.phase))
        if self.start_zero:
            xacc[t < 0] = 0
        return xacc

    def yacc(self, t):
        yacc = - self.max_speed*self.w*(np.cos(t*self.w)*np.cos(self.A*np.cos(t*self.w)+self.phase)
                                        + self.A*np.sin(t*self.w)**2*np.sin(self.A*np.cos(t*self.w)+self.phase))
        if self.start_zero:
            yacc[t < 0] = 0
        return yacc

    def zacc(self, t):
        return 0

    def get_period(self):
        return 2*np.pi/self.w


class LinearAcceleratingCharge(Charge):
    """Point charge accelerates in x direction starting at origin."""

    def __init__(self, pos_charge=True, acceleration=0.1*c, stop_t=None):
        super().__init__(pos_charge)
        self.acceleration = acceleration
        if stop_t is None:
            self.stop_t = 0.9999*c/acceleration  # so v is never greater than c
        else:
            self.stop_t = stop_t

    def xpos(self, t):
        xpos = 0.5*self.acceleration*t**2
        xpos[t < 0] = 0
        xpos[t > self.stop_t] = 0.5*self.acceleration*self.stop_t**2 + \
            self.acceleration*self.stop_t * (t[t > self.stop_t]-self.stop_t)
        return xpos

    def ypos(self, t):
        return 0

    def zpos(self, t):
        return 0

    def xvel(self, t):
        xvel = self.acceleration*t
        xvel[t < 0] = 0
        xvel[t > self.stop_t] = self.acceleration*self.stop_t
        return xvel

    def yvel(self, t):
        return 0

    def zvel(self, t):
        return 0

    def xacc(self, t):
        xacc = self.acceleration*np.ones(t.shape)
        xacc[t < 0] = 0
        xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t):
        return 0

    def zacc(self, t):
        return 0


class LinearDeceleratingCharge(Charge):
    """Point charge decelerates in x direction starting at origin.."""

    def __init__(self, pos_charge=True, deceleration=0.1*c, initial_speed=0.999*c, stop_t=None):
        super().__init__(pos_charge)
        self.deceleration = deceleration
        self.initial_speed = initial_speed
        if stop_t is None:
            self.stop_t = initial_speed/deceleration  # so v is zero at stop
        else:
            self.stop_t = stop_t

    def xpos(self, t):
        xpos = self.initial_speed*t
        xpos[t > 0] = self.initial_speed*t[t > 0] - \
            0.5*self.deceleration*t[t > 0]**2
        xpos[t > self.stop_t] = self.initial_speed * \
            self.stop_t - 0.5*self.deceleration*self.stop_t**2
        return xpos

    def ypos(self, t):
        return 0

    def zpos(self, t):
        return 0

    def xvel(self, t):
        xvel = self.initial_speed*np.ones(t.shape)
        xvel[t > 0] = self.initial_speed - self.deceleration*t[t > 0]
        xvel[t > self.stop_t] = 0
        return xvel

    def yvel(self, t):
        return 0

    def zvel(self, t):
        return 0

    def xacc(self, t):
        xacc = np.zeros(t.shape)
        xacc[t > 0] = -self.deceleration
        xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t):
        return 0

    def zacc(self, t):
        return 0


class LinearVelocityCharge(Charge):
    """Point charge decelerates in x direction starting at origin.."""

    def __init__(self, pos_charge=True, speed=0.99*c, init_pos=0):
        super().__init__(pos_charge)
        self.speed = speed
        self.init_pos = init_pos

    def xpos(self, t):
        return self.speed*t + self.init_pos

    def ypos(self, t):
        return 0

    def zpos(self, t):
        return 0

    def xvel(self, t):
        return self.speed

    def yvel(self, t):
        return 0

    def zvel(self, t):
        return 0

    def xacc(self, t):
        return 0

    def yacc(self, t):
        return 0

    def zacc(self, t):
        return 0
