import matplotlib.pyplot as plt
import random
import numpy as np

title_font_size = 25 
label_size = 16


def random_walk_2d(steps, random_function = random.random):
    x, y = 0, 0
    x_coords, y_coords = [x], [y]

    for _ in range(steps):
        direction = int(random_function() * 4)
        if direction == 0:
            x += 1  # Move right
        elif direction == 1:
            x -= 1  # Move left
        elif direction == 2:
            y += 1  # Move up
        else:
            y -= 1  # Move down

        x_coords.append(x)
        y_coords.append(y)

    return x_coords, y_coords

def self_avoiding_random_walk(steps):
    terminated = False
    x, y = 0, 0
    visited = {(x, y)}  # Set to store visited coordinates
    last_direction = None
    x_coords, y_coords = [x], [y]  # Lists to store the coordinates of the walk

    for _ in range(steps):
        possible_directions = {0, 1, 2, 3} - {last_direction}
        #print(possible_directions)
        if not possible_directions:
            break  # No more possible directions

        direction = random.choice(list(possible_directions))

        if direction == 0:
            last_direction = 1
            x += 1
        elif direction == 1:
            last_direction = 0
            x -= 1
        elif direction == 2:
            last_direction = 3
            y += 1
        else:
            last_direction = 2
            y -= 1

        if (x, y) in visited:
            # i want to se it intersect
            x_coords.append(x)
            y_coords.append(y)
            terminated = True
            break  # Self-crossing, terminate walk
        visited.add((x, y))
        x_coords.append(x)
        y_coords.append(y)

    return x_coords, y_coords, terminated 

def random_walk_2d_own(steps,a,c,m):
    x, y = 0, 0
    x_coords, y_coords = [x], [y]
    # previous r in the sequence
    r_prev = 1
    

    for _ in range(steps):
        r = custom_random_generator(r_prev, a, c, m)
        r_prev = r
        direction = r//(m//4)
        if direction == 0:
            x += 1  # Move right
        elif direction == 1:
            x -= 1  # Move left
        elif direction == 2:
            y += 1  # Move up
        else:
            y -= 1  # Move down

        x_coords.append(x)
        y_coords.append(y)

    return x_coords, y_coords

def custom_random_generator(r_prev, a=3, c=4, m=128):
    """
    Custom random number generator.
    
    Args:
    r_prev (int): The previous random number.
    a (int): The multiplier parameter.
    c (int): The increment parameter.
    m (int): The modulus parameter.

    Returns:
    int: The next random number in the sequence.
    """

    return (a * r_prev + c) % m 

def calculate_r(x_coords, y_coords):
    return np.sqrt(x_coords[-1]**2 + y_coords[-1]**2)




def ex2a(): 
    for steps in [10, 100, 1000]:
        x_coords, y_coords = random_walk_2d(steps)
        plt.figure(figsize=(8, 8))
        plt.plot(x_coords, y_coords, marker='o')
        plt.title(f"2D Random Walk with {steps} Steps", fontsize=title_font_size)
        plt.xlabel("X", fontsize=label_size)
        plt.ylabel("Y", fontsize=label_size)
        #plt.grid(True)
        # Save the plot with a descriptive filename
        filename = f"2D_Random_Walk_{steps}_Steps.png"
        plt.savefig(filename)
        plt.show()

def ex2b(): 
    for steps in [10, 100, 1000]:
        m = 129
        a = 5
        c = 90

        x_coords, y_coords = random_walk_2d_own(steps,a,c,m)
        plt.figure(figsize=(8, 8))
        plt.plot(x_coords, y_coords, marker='o')
        plt.title(f"2D Pseudo Random Walk with {steps} Steps", fontsize=title_font_size)
        plt.xlabel("X", fontsize=label_size)
        plt.ylabel("Y", fontsize=label_size)
        #plt.grid(True)
        # Save the plot with a descriptive filename
        filename = f"{steps}stepsa{a}c{c}m{m}.png"
        plt.savefig(filename)
        plt.show()
    
def ex2c(): 
    # Store RMS values for different step counts
    rms_values = []
    fluctuations = []
    stderrors = []
    step_counts = range(1, 1001, 10)  # Adjust the step for finer or coarser granularity
    walks = 100

    for steps in step_counts:
        distances = []
        
        for _ in range(walks):  # Number of random walks to average over
            x_coords, y_coords = random_walk_2d(steps)
            distances.append(calculate_r(x_coords, y_coords))

        distances = np.array(distances)

        rms = np.sqrt(np.mean(distances**2))
        fluct = np.sqrt((np.mean(distances**2) - np.mean(distances)**2) *  walks / (walks -1))
        std = np.sqrt((np.mean(distances**2) - np.mean(distances)**2) / (walks -1))
    
        #rms_mean = np.mean(distances)
        rms_values.append(rms)
        fluctuations.append(fluct)
        stderrors.append(std)

    # Plotting RMS values
    plt.figure(figsize=(10, 6))
    plt.plot(step_counts, rms_values, marker='o', label='RMS End-to-End Distance')
    plt.plot(step_counts, fluctuations, marker='x', label='Root Mean Squared Fluctuation')
    plt.plot(step_counts, stderrors, label='Standard error')
    plt.title("RMS End-to-End Distance and Fluctuation vs. Number of Steps", fontsize=16)
    plt.xlabel("Number of Steps")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

def self_cross_plot(plot_walks = True): 
    # only used to sense check the validity of the self_avoiding_random_walk function
    for steps in [10, 100]:
        terminated = True
        while terminated: 
            x_coords, y_coords, terminated = self_avoiding_random_walk(steps)

        if plot_walks: 
            plt.figure(figsize=(8, 8))
            plt.plot(x_coords, y_coords, marker='o')
            plt.title(f"2D Random Walk with {steps} Steps", fontsize=title_font_size)
            plt.xlabel("X", fontsize=label_size)
            plt.ylabel("Y", fontsize=label_size)
            #plt.grid(True)
            # Save the plot with a descriptive filename
            filename = f"2D_Random_Walk_{steps}_Steps.png"
            plt.savefig(filename)
            plt.show()

def ex2d():
    max_steps = 100
    trials_per_step = 100

    termination_fractions = []
    step_counts = range(1, max_steps + 1)

    for steps in step_counts:
        terminated_count = 0
        for _ in range(trials_per_step):
            _, _, terminated = self_avoiding_random_walk(steps)
            if terminated:
                terminated_count += 1
        fraction = terminated_count / trials_per_step
        termination_fractions.append(fraction)

    # Plotting the termination fractions
    plt.figure(figsize=(10, 6))
    plt.plot(step_counts, termination_fractions, marker='o')
    plt.title("Fraction of Terminated Self-Avoiding Random Walks vs. Number of Steps", fontsize=16)
    plt.xlabel("Number of Steps")
    plt.ylabel("Fraction of Terminated Walks")
    plt.grid(True)
    plt.show()

def ex2e(): 
    # Store RMS values for different step counts
    rms_values = []
    fluctuations = []
    inter_rms_values = []   # inter are those that allow intersections
    inter_fluctuations = []

    step_counts = range(1, 61, 1)  # Adjust the step for finer or coarser granularity
    walks = 100

    for steps in step_counts:
        inter_distances = []
        distances = []
        
        while len(distances) < walks:  # Number of random walks to average over
            x_coords, y_coords, terminated = self_avoiding_random_walk(steps)
            if len(inter_distances) < walks: # dont need more than 100 distances in the intersecting one
                i_x_coords, i_y_coords = random_walk_2d(steps)
                inter_distances.append(calculate_r(i_x_coords, i_y_coords))

            # if it's been terminated we can't add that value to the distances array
            if not terminated: 
                distances.append(calculate_r(x_coords, y_coords))

        distances = np.array(distances)
        inter_distances = np.array(inter_distances)

        rms = np.sqrt(np.mean(distances**2))
        fluct = np.sqrt((np.mean(distances**2) - np.mean(distances)**2) *  walks / (walks -1))

        i_rms = np.sqrt(np.mean(inter_distances**2))
        i_fluct = np.sqrt((np.mean(inter_distances**2) - np.mean(inter_distances)**2) *  walks / (walks -1))

    
        #rms_mean = np.mean(distances)
        rms_values.append(rms)
        fluctuations.append(fluct)
        inter_rms_values.append(i_rms)
        inter_fluctuations.append(i_fluct)
        
    
    # Plotting RMS values
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(step_counts), np.log(rms_values), marker='o', label='RMS End-to-End Distance')
    plt.plot(np.log(step_counts), np.log(inter_rms_values), marker='o', label='RMS End-to-End Distance with intersections')
    plt.title("RMS End-to-End Distance and Fluctuation vs. Number of Steps", fontsize=16)
    plt.xlabel("Natural logarithm of Number of Steps")
    plt.ylabel("Natural logarithm of Distance")
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__": 
    ex2e()
    

