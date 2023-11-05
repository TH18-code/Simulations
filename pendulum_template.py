#!/bin/python3

# Python simulation of a simple planar pendulum with real time animation
# BH, OF, MP, AJ, TS 2020-10-20, latest version 2022-10-25.

from matplotlib import animation
from pylab import *

"""
    This script defines all the classes needed to simulate (and animate) a single pendulum.
    Hierarchy (somehow in order of encapsulation):
    - Oscillator: a struct that stores the parameters of an oscillator (harmonic or pendulum)
    - Observable: a struct that stores the oscillator's coordinates and energy values over time
    - BaseSystem: harmonic oscillators and pendolums are distinguished only by the expression of
                    the return force. This base class defines a virtual force method, which is
                    specified by its child classes
                    -> Harmonic: specifies the return force as -k*t (i.e. spring)
                    -> Pendulum: specifies the return force as -k*sin(t)
    - BaseIntegrator: parent class for all time-marching schemes; function integrate performs
                    a numerical integration steps and updates the quantity of the system provided
                    as input; function timestep wraps the numerical scheme itself and it's not
                    directly implemented by BaseIntegrator, you need to implement it in his child
                    classes (names are self-explanatory)
                    -> EulerCromerIntegrator: ...
                    -> VerletIntegrator: ...
                    -> RK4Integrator: ...
    - Simulation: this last class encapsulates the whole simulation procedure; functions are 
                    self-explanatory; you can decide whether to just run the simulation or to
                    run while also producing an animation: the latter option is slower
"""

# Global constants
G = 9.8  # gravitational acceleration

class Oscillator:

    """ Class for a general, simple oscillator """

    def __init__(self, m=1, c=4, t0=0, theta0=0, dtheta0=0, gamma=0):
        self.m = m              # mass of the pendulum bob
        self.c = c              # c = g/L
        self.L = G / c          # string length
        self.t = t0             # the time
        self.theta = theta0     # the position/angle
        self.dtheta = dtheta0   # the velocity
        self.gamma = gamma      # damping

class Observables:

    """ Class for storing observables for an oscillator """

    def __init__(self):
        self.time = []          # list to store time
        self.pos = []           # list to store positions
        self.vel = []           # list to store velocities
        self.energy = []        # list to store energy


class BaseSystem:
    
    def force(self, osc):

        """ Virtual method: implemented by the childc lasses  """

        pass

    def force_spec(self,theta, dtheta, osc): 
        """ Virtual method: implemented by the childc lasses  """

        pass


class Harmonic(BaseSystem):
    def force(self, osc):
        return - osc.m * ( osc.c*osc.theta + osc.gamma*osc.dtheta )

    def force_spec(self,theta, dtheta, osc): 
        return - osc.m * (osc.c*theta + osc.gamma*dtheta )
        
        
class Pendulum(BaseSystem):
    def force(self, osc):
        return - osc.m * ( osc.c*np.sin(osc.theta) + osc.gamma*osc.dtheta )
    
    def force_spec(self,theta, dtheta, osc): 
        return - osc.m * ( osc.c*np.sin(theta) + osc.gamma*dtheta )


class BaseIntegrator:

    def __init__(self, _dt=0.01) :
        self.dt = _dt   # time step

    def integrate(self, simsystem, osc, obs):

        """ Perform a single integration step """
        
        self.timestep(simsystem, osc, obs)

        # Append observables to their lists
        obs.time.append(osc.t)
        obs.pos.append(osc.theta)
        obs.vel.append(osc.dtheta)
        # Function 'isinstance' is used to check if the instance of the system object is 'Harmonic' or 'Pendulum'
        if isinstance(simsystem, Harmonic) :
            # Harmonic oscillator energy
            obs.energy.append(0.5 * osc.m * osc.L ** 2 * osc.dtheta ** 2 + 0.5 * osc.m * G * osc.L * osc.theta ** 2)
        else :
            # Pendulum energy
            obs.energy.append(0.5 * osc.m * osc.L ** 2 * osc.dtheta ** 2 + osc.m * G * osc.L * (1 - np.cos(osc.theta))) 
            # TODO: Append the total energy for the pendulum (use the correct formula!)


    def timestep(self, simsystem, osc, obs):

        """ Virtual method: implemented by the child classes """
        
        pass


# HERE YOU ARE ASKED TO IMPLEMENT THE NUMERICAL TIME-MARCHING SCHEMES:

class EulerCromerIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        osc.dtheta += accel * self.dt
        osc.theta += osc.dtheta * self.dt


class VerletIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        osc.theta += osc.dtheta * self.dt + 0.5*accel*(self.dt**2)
        # need to calculate acceleration in new position aswell 
        # assumes accel is not a function of speed (correct as long as dampening i zero, wrong otherwise)
        # best practice would be to use symplectic or newtons method to get v(t + dt) as a(t+dt) depends on v(t+dt)
        acceldt = simsystem.force(osc) / osc.m
        osc.dtheta += 0.5*(acceldt + accel)*self.dt

class RK4Integrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m 
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        theta = osc.theta 
        dtheta = osc.dtheta

        a1 = accel * self.dt
        b1 = osc.dtheta * self.dt

        a2 = (simsystem.force_spec(theta + 0.5 * b1, dtheta + 0.5 * a1, osc) / osc.m) * self.dt
        b2 = (dtheta + 0.5 * a1) * self.dt

        a3 = (simsystem.force_spec(theta + 0.5 * b2, dtheta + 0.5 * a2, osc) / osc.m) * self.dt
        b3 = (dtheta + 0.5 * a2) * self.dt

        a4 = (simsystem.force_spec(theta + 0.5 * b3, dtheta + 0.5 * a3, osc) / osc.m) * self.dt
        b4 = (dtheta + 0.5 * a3) * self.dt

        osc.dtheta += (1/6) * (a1 + 2 * a2 + 2 * a3 + a4)
        osc.theta += (1/6) * (b1 + 2 * b2 + 2 * b3 + b4)
        



# Animation function which integrates a few steps and return a line for the pendulum
def animate(framenr, simsystem, oscillator, obs, integrator, pendulum_line, stepsperframe):
    
    for it in range(stepsperframe):
        integrator.integrate(simsystem, oscillator, obs)

    x = np.array([0, np.sin(oscillator.theta)])
    y = np.array([0, -np.cos(oscillator.theta)])
    pendulum_line.set_data(x, y)
    return pendulum_line,


class Simulation:

    def reset(self, osc=Oscillator()) :
        self.oscillator = osc
        self.obs = Observables()

    def __init__(self, osc=Oscillator()) :
        self.reset(osc)

    # Run without displaying any animation (fast)
    def run(self,
            simsystem,
            integrator,
            tmax=30.,               # final time
            ):

        n = int(tmax / integrator.dt)

        for it in range(n):
            integrator.integrate(simsystem, self.oscillator, self.obs)

    # Run while displaying the animation of a pendulum swinging back and forth (slow-ish)
    # If too slow, try to increase stepsperframe
    def run_animate(self,
            simsystem,
            integrator,
            tmax=30.,               # final time
            stepsperframe=1         # how many integration steps between visualising frames
            ):

        numframes = int(tmax / (stepsperframe * integrator.dt))-2

        # WARNING! If you experience problems visualizing the animation try to comment/uncomment this line
        plt.clf()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # fig = plt.figure()

        ax = plt.subplot(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        plt.axhline(y=0)  # draw a default hline at y=1 that spans the xrange
        plt.axvline(x=0)  # draw a default vline at x=1 that spans the yrange
        pendulum_line, = ax.plot([], [], lw=5)
        plt.title(title)
        # Call the animator, blit=True means only re-draw parts that have changed
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[simsystem,self.oscillator,self.obs,integrator,pendulum_line,stepsperframe],
                                       frames=numframes, interval=25, blit=True, repeat=False)

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # plt.show()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        plt.waitforbuttonpress(10)

    # Plot coordinates and energies (to be called after running)
    def plot_observables(self, integrator, title="simulation", ref_E=None):

        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.pos, 'b-', label="Position")
        plt.plot(self.obs.time, self.obs.vel, 'r-', label="Velocity")
        plt.plot(self.obs.time, self.obs.energy, 'g-', label="Energy")
        if ref_E != None :
            plt.plot([self.obs.time[0],self.obs.time[-1]] , [ref_E, ref_E], 'k--', label="Ref.")
        plt.xlabel('time')
        plt.ylabel('observables')
        plt.legend()
        plt.savefig("0.1" + "verharm" + str(integrator.dt) + ".png")
        plt.show()
    
    def plotharmonic(self, theta0, osc): 
        # assumes dtheta0 = 0 
        plt.clf()
        plt.title("Harmonic exact solution")
        t = np.array(self.obs.time)
        pos = theta0*np.cos((osc.c**0.5)*t)
        vel = -(osc.c**0.5)*theta0*np.sin((osc.c**0.5)*t)
        # 0.5 * osc.m * osc.L ** 2 * osc.dtheta ** 2 + 0.5 * osc.m * G * osc.L * osc.theta ** 2
        E = 0.5 * osc.m * osc.L ** 2 * (vel ** 2) + 0.5 * osc.m * G * osc.L * (pos ** 2)
        plt.plot(t, pos, 'b-', label = "Postion")
        plt.plot(t, vel, 'r-', label="Velocity")
        plt.plot(t,E, 'g-', label="Energy")
        plt.xlabel('time')
        plt.ylabel('observables')
        plt.legend()

        plt.show()

    def get_period(self): 
        obs = self.obs
        # period is the time it takes to go from positive to negative to positive to negative
        s_changes = 0 
        t = []

        s = sign(self.obs.pos[0])

        for i in range(len(self.obs.pos)): 
            si = sign(self.obs.pos[i])
            if si != s: 
                s = si
                s_changes +=1 
                t.append(self.obs.time[i])
            if s_changes == 3: 
                break 
         
        return t[-1] - t[0]
  




# It's good practice to encapsulate the script execution in 
# a function (e.g. for profiling reasons)
def exercise_11() :
    oscillator = Oscillator(theta0=np.pi * 0.1)
    sim = Simulation(oscillator)
    simsystem = Harmonic()
    #integrator = EulerCromerIntegrator()
    integrator = VerletIntegrator(_dt = 0.01)
    #integrator = RK4Integrator()
    sim.run(simsystem, integrator)
    sim.plot_observables(integrator)
    t = sim.get_period()
    print(t)

    oscillator = Oscillator(theta0=np.pi * 0.1)
    sim = Simulation(oscillator)
    simsystem = Harmonic()
    #integrator = EulerCromerIntegrator()
    integrator = VerletIntegrator(_dt = 0.1)
    #integrator = RK4Integrator()
    sim.run(simsystem, integrator)
    sim.plot_observables(integrator)

    oscillator = Oscillator(theta0=np.pi * 0.1)
    sim = Simulation(oscillator)
    simsystem = Harmonic()
    #integrator = EulerCromerIntegrator()
    integrator = VerletIntegrator(_dt = 0.5)
    #integrator = RK4Integrator()
    sim.run(simsystem, integrator)
    sim.plot_observables(integrator)


    sim.plotharmonic(np.pi *0.1, oscillator)



    # TODO


def exercise_12(): 
    oscillator = Oscillator(theta0=np.pi * 0.1)
    sim = Simulation(oscillator)
    simsystem = Pendulum()  
    integrator = VerletIntegrator(_dt = 0.01)
    sim.run(simsystem, integrator)
    T = []
   
   
    theta0s = np.arange(start = 0.001, stop = np.pi * 0.7, step = 0.1)
    for theta in theta0s: 
         oscillator = Oscillator(theta0=theta)
         sim.reset(oscillator)
         sim.run(simsystem, integrator)
         T.append(sim.get_period())
    
    perturbs = np.pi*(1 + (1/16)*(theta0s**2) + (11/3072) * (theta0s**4) + (173/737280)*(theta0s**6))
    
    plt.clf() 
    plt.title("Periodtime as a function of intitial values")
    plt.plot(theta0s, T, 'b-', label = "Period")
    plt.plot(theta0s, perturbs, 'g-', label = "Perturbation")
    plt.xlabel("Intital angle $\Theta$")
    plt.ylabel("Periodtime T")
    plt.legend()
    plt.show()




    return 


"""
    This directive instructs Python to run what comes after ' if __name__ == "__main__" : '
    if the script pendulum_template.py is executed 
    (e.g. by running "python3 pendulum_template.py" in your favourite terminal).
    Otherwise, if pendulum_template.py is imported as a library 
    (e.g. by calling "import pendulum_template as dp" in another Python script),
    the following is ignored.
    In this way you can choose whether to code the solution to the exericises here in this script 
    or to have (a) separate script(s) that include pendulum_template.py as library.
"""
if __name__ == "__main__" :
    #exercise_11()
    exercise_12()
    # ...
