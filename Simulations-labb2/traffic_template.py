#!/bin/python3

# Template for traffic simulation
# BH, MP 2021-11-15, latest version 2022-11-1.

"""
    This template is used as backbone for the traffic simulations.
    Its structure resembles the one of the pendulum project, that is you have:
    (a) a class containing the state of the system and it's parameters
    (b) a class storing the observables that you want then to plot
    (c) a class that propagates the state in time (which in this case is discrete), and
    (d) a class that encapsulates the aforementioned ones and performs the actual simulation
    You are asked to implement the propagation rule(s) corresponding to the traffic model(s) of the project.
"""

import math
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy.random as rng
import numpy as np

import matplotlib


class Cars:

    """ Class for the state of a number of cars """

    def __init__(self, numCars=5, roadLength=50, v0=1):
        self.numCars    = numCars
        self.roadLength = roadLength
        self.t  = 0
        self.x  = []
        self.v  = []
        self.c  = []      # seems to be color
        for i in range(numCars):

            # TODO: Set the initial position for each car.
            # Note that the ordering of the cars on the road needs to match
            # the order in which you compute the distances between cars
            self.x.append(i)         # the position of the cars on the road
            self.v.append(v0)       # the speed of the cars
            self.c.append(i)        # the color of the cars (for drawing)

    # NOTE: you can, but don't have to use this function for computing distances
    def distance(self, i):
        # TODO: Implement the function returning the PERIODIC distance 
        # between car i and the one in front 
        return (self.x[(i+1)%self.numCars] - self.x[i])%self.roadLength   # i tweeked it a bit so that it becomes periodic


class Observables:

    """ Class for storing observables """

    def __init__(self):
        self.time = []          # list to store time
        self.flowrate = []      # list to store the flow rate
        self.history = [] # contains all historic positions

        

class BasePropagator:

    def __init__(self):
        return
        
    def propagate(self, cars, obs):

        """ Perform a single integration step """
        
        fr = self.timestep(cars, obs)

        # Append observables to their lists
        obs.time.append(cars.t) # where should the time be updated?
        obs.flowrate.append(fr)
        obs.history.append(cars.x.copy())
        

              
    def timestep(self, cars, obs):

        """ Virtual method: implemented by the child classes """
        
        pass
      
        
class ConstantPropagator(BasePropagator) :
    
    """ 
        Cars do not interact: each position is just 
        updated using the corresponding velocity 
    """
    
    def timestep(self, cars, obs):
        for i in range(cars.numCars):
            cars.x[i] += cars.v[i]
        cars.t += 1
        return 0

class MyPropagator(BasePropagator) :

    def __init__(self, vmax, p):
        BasePropagator.__init__(self)
        self.vmax = vmax
        self.p = p

    def timestep(self, cars, obs):

        # TODO Here you should implement the car behaviour rules

        for i in range(cars.numCars): 
            # increase velocity 
            if cars.v[i] < self.vmax: 
                cars.v[i] += 1
        
        for i in range(cars.numCars): 
            # copmute distances and reduce velocities
            d = cars.distance(i)
            if cars.v[i] >= d: 
                cars.v[i] = d - 1
        
        for i in range(cars.numCars):
            # probability decrease speed  
            
            r = rng.binomial(1, self.p)
            
            if r and cars.v[i] > 0: 
                cars.v[i] -= 1

        for i in range(cars.numCars): 
            # update all positions 
            cars.x[i] += cars.v[i]
            
        v = np.array(cars.v)
        flowrate = np.sum(v) / cars.roadLength
        
        cars.t += 1
        return flowrate
        


############################################################################################

def draw_cars(cars, cars_drawing):

    """ Used later on to generate the animation """
    theta = []
    r     = []

    for position in cars.x:
        # Convert to radians for plotting  only (do not use radians for the simulation!)
        theta.append(position * 2 * math.pi / cars.roadLength)
        r.append(1)

    return cars_drawing.scatter(theta, r, c=cars.c, cmap='hsv')


def animate(framenr, cars, obs, propagator, road_drawing, stepsperframe):

    """ Animation function which integrates a few steps and return a drawing """

    for it in range(stepsperframe):
        propagator.propagate(cars, obs)

    return draw_cars(cars, road_drawing),


class Simulation:

    def reset(self, cars=Cars()) :
        self.cars = cars
        self.obs = Observables()

    def __init__(self, cars=Cars()) :
        self.reset(cars)

    def plot_history(self, title="simulation"): 
        plt.clf()
        plt.title(title)
        self.obs.history = np.array(self.obs.history).transpose()
        for i in range(self.cars.numCars): 
            plt.plot(self.obs.time, self.obs.history[i][:])
        plt.xlabel('time')
        plt.ylabel('position')
        plt.savefig(title + ".pdf")
        plt.show()


    def plot_observables(self, title="simulation"):

        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.flowrate)
        plt.xlabel('time')
        plt.ylabel('flow rate')
        plt.savefig(title + ".pdf")
        plt.show()

    # Run without displaying any animation (fast)
    def run(self,propagator,
            numsteps=200,           # final time
            title="simulation",     # Name of output file and title shown at the top
            ):

        for it in range(numsteps):
            propagator.propagate(self.cars, self.obs)

        #self.plot_observables(title)

    # Run while displaying the animation of bunch of cars going in circe (slow-ish)
    def run_animate(self,
            propagator,
            numsteps=200,           # Final time
            stepsperframe=1,        # How many integration steps between visualising frames
            title="simulation",     # Name of output file and title shown at the top
            ):

        numframes = int(numsteps / stepsperframe)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.axis('off')
        # Call the animator, blit=False means re-draw everything
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[self.cars,self.obs,propagator,ax,stepsperframe],
                                       frames=numframes, interval=50, blit=True, repeat=False)
        plt.show()

        # If you experience problems visualizing the animation and/or
        # the following figures comment out the next line 
        # plt.waitforbuttonpress(30)

        #self.plot_observables(title)


# It's good practice to encapsulate the script execution in 
# a main() function (e.g. for profiling reasons)
def main() :

    # Here you can define one or more instances of cars, with possibly different parameters, 
    # and pass them to the simulator 

    # Be sure you are passing the correct initial conditions!
    cars = Cars(numCars = 30, roadLength=50)

    # Create the simulation object for your cars instance:
    simulation = Simulation(cars)

    # simulation.run_animate(propagator=ConstantPropagator())
    simulation.run_animate(propagator=MyPropagator(vmax=2, p=0.5))

def ex2a(): 
    road_length = 40
    densities = np.linspace(0.1, 1, 500)  # Different densities from 0.1 to 1
    average_flow_rates = []

    for density in densities:
        num_cars = int(road_length * density)
        cars = Cars(numCars=num_cars, roadLength=road_length)
        simulation = Simulation(cars)
        propagator = MyPropagator(vmax=2, p=0.5)
        simulation.run(propagator=propagator, numsteps=1000)
        
        # Calculate average flow rate after the system has equilibrated
        equilibration_time = 100  # Assuming equilibration time
        average_flow_rate = np.mean(simulation.obs.flowrate[equilibration_time:])
        average_flow_rates.append(average_flow_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, average_flow_rates, marker='o')
    plt.title("Average Flow Rate vs Car Density")
    plt.xlabel("Car Density")
    plt.ylabel("Average Flow Rate")
    plt.grid(True)
    plt.show()

def run_simulation(num_cars, road_length, vmax, p, num_steps):
    cars = Cars(numCars=num_cars, roadLength=road_length)
    simulation = Simulation(cars)
    propagator = MyPropagator(vmax=vmax, p=p)
    simulation.run(propagator=propagator, numsteps=num_steps)
    #simulation.plot_history()
    return simulation.obs.flowrate  

def estimate_statistical_accuracy(target_standard_error, max_simulations=1000):
    road_length = 50
    num_cars = 25
    vmax = 2
    p = 0.5
    num_steps = 500  # Including time for equilibration
    flow_rates = []
    standard_errors = []

    for _ in range(max_simulations):
        n = run_simulation(num_cars, road_length, vmax, p, num_steps)
        flow_rate = np.mean(n[-100:])
        flow_rates.append(flow_rate)
        current_standard_error = np.std(flow_rates)/ np.sqrt(len(flow_rates) - 1)   # becomes the same formula
        standard_errors.append(current_standard_error)
        if current_standard_error < target_standard_error:
            break

    return flow_rates, standard_errors, current_standard_error

def ex2b(): 
    target_standard_error = 0.001
    flow_rates, standard_errors, achieved_standard_error = estimate_statistical_accuracy(target_standard_error)
    print("Achieved Standard Error:", achieved_standard_error)
    print("Number of Simulations:", len(flow_rates))

    plt.figure(figsize=(10, 6))
    x = range(1, len(flow_rates) + 1)
    plt.plot(x, standard_errors, marker='o')
    plt.title("Standarderror over amount of simulations")
    plt.xlabel("Simulations")
    plt.ylabel("Standard error")
    plt.grid(True)
    plt.show()


def fundamental_diagram(road_length, densities, v=2, p=0.5): 
    # returns the average flowrates for different densities, which makes it easy to plot the fundamental diagram
    average_flow_rates = []

    for density in densities:
        num_cars = int(road_length * density)
        cars = Cars(numCars=num_cars, roadLength=road_length)
        simulation = Simulation(cars)
        propagator = MyPropagator(vmax=v, p=p)  # these used to be vmax = 2, p = 0.5
        simulation.run(propagator=propagator, numsteps=1000)
        
        # Calculate average flow rate after the system has equilibrated
        equilibration_time = 100  # Assuming equilibration time
        average_flow_rate = np.mean(simulation.obs.flowrate[-300:]) # this used to be indexed equilibration_time:
        average_flow_rates.append(average_flow_rate)

    return average_flow_rates

def ex2c(): # denna blir fel just nu, fattar inte vad som inte funkar. 
    plt.figure(figsize=(10, 6))
    lenghts = np.linspace(1, 100, 5)
    densities = np.linspace(0.1, 1, 50) 

    for l in lenghts: 
        print(l)
        mean_flow_rates = fundamental_diagram(l, densities)

        plt.plot(densities, mean_flow_rates)
    
    plt.show()
        


def plot_fundamental_diagrams(lengths=[20, 40, 60, 80, 100,150, 200, 250, 300]):
    densities = np.linspace(0.1, 1, 50)  # Common densities for all lengths

    for road_length in lengths:
        average_flow_rates = []

        for density in densities:
            num_cars = int(road_length * density)
            cars = Cars(numCars=num_cars, roadLength=road_length)
            simulation = Simulation(cars)
            propagator = MyPropagator(vmax=2, p=0.5)
            simulation.run(propagator=propagator, numsteps=1000)

            # Calculate and append the average flow rate
            # Assuming you have a way to calculate this in your code
            average_flow_rate = np.mean(simulation.obs.flowrate[-100:]) # ändra denna till fler steg bakåt för att få mindre fluktuationer


            average_flow_rates.append(average_flow_rate)

        plt.plot(densities, average_flow_rates, label=f'Length {road_length}')

    plt.xlabel('Density')
    plt.ylabel('Flow')
    plt.title('Fundamental Diagrams for Different Road Lengths')
    plt.legend()
    plt.show()




def ex2d(): 
    # can also be modified for task e
    densities = np.linspace(0.1, 1, 100)  # Common densities for all lengths
    velocities = [1,2,5]
    probs = [0.2,0.5,0.8]
    
    for v in velocities: 
        mean_flow_rates = fundamental_diagram(200, densities, v=v, p=0.5)

        plt.plot(densities, mean_flow_rates, label=f'v_max = {v}')

    plt.xlabel('Density')
    plt.ylabel('Flow')
    plt.legend()
    plt.show()





# Calling 'main()' if the script is executed.
# If the script is instead just imported, main is not called (this can be useful if you want to
# write another script importing and utilizing the functions and classes defined in this one)
if __name__ == "__main__" :
    #run_simulation(10, 50, 2, 0.5, 50)
    ex2d()
    #plot_fundamental_diagrams()

"""om komplettering på det där med många fundamentaldiagram 
runna 
plot_fundemental_diagrams()

gå in i fuktionen och gör så att man tar medelvärdet över en mkt längre tid så borde det nog bli en mkt 
jämnare graf:) 
"""