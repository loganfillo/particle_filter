import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.stats import norm 

def prediction(particles, command):
    """
    Given the previous particle set, produces a new particle set containing the predicted
    poses of each particle using the proposal distribution. The particle weight can be ignored for now

    @param particles: A (n_particles, 4) numpy array where each row is a particle consiting
    of [x,y,theta,w] where x is x position, y is y position, theta is rotation in radians, and
    w is the weight of the particle

    @param command: A tuple (dist, angle) where dist is the distance the robot is command to travel
    during this iteration (in the direction of it's body-fixed frame), and angle is the angle the robot
    is command to rotate during this iteration

    @returns A new particle set of size (n_particles, 4) where each particle from the input set has a new
    pose as predicted from the proposal distribution. 
    """
    n_particles,_  = particles.shape
    particles[:,2] += (command[1] + np.random.randn(n_particles)*0.01)
    particles[:,2] %= 2*np.pi
    particles[:,0] += (command[0] * np.cos(particles[:,2]) + np.random.randn(n_particles))
    particles[:,1] += (command[0] * np.sin(particles[:,2]) + np.random.randn(n_particles))
    return particles

def compute_weights(particles, marker_pos, x_i, sensor_noise):
    """
    Computes the importance weights for each pose hypothesis in the input particle set. Normalizes
    the weights so that they sum to 1

    @param particles: A (n_particles, 4) numpy array where each row is a particle consiting
    of [x,y,theta,w] where x is x position, y is y position, theta is rotation in radians, and
    w is the weight of the particle

    @param marker_pos: A (n_markers, 2) numpy array where each entry i is the (x,y) position
    of the marker i

    @param x_i: A (n_markers,1) nump array where each entry i is the observed distance of the
    robot to the marker i

    @param sensor_noise: A number representing sensor noise, this is used at the std dev when
    we compute p(x_i | z_i = z^k_i)

    @returns The prediciton particle set containing both pose hypotheses and normalized weights
    """
    n_particles,_  = particles.shape
    weights = np.ones((n_particles,))
    for j, marker in enumerate(marker_pos):
        dist = np.linalg.norm(particles[:,:2]-marker, axis=1)
        weights *= norm(dist, sensor_noise).pdf(x_i[j])
    weights += 1.e-300
    particles[:,3] = weights/np.sum(weights)
    return particles

def resample(particles):
    """
    Resample the particles using their importance weights. Returns the new set

    @param particles: A (n_particles, 4) numpy array where each row is a particle consiting
    of [x,y,theta,w] where x is x position, y is y position, theta is rotation in radians, and
    w is the weight of the particle

    @returns A new particle set of size (n_particles, 4) of newly sampled particles 
    """
    n_particles,_  = particles.shape
    if (1. / np.sum(np.square(particles[:,3]))) < (n_particles/2):
        cumulative_sum = np.cumsum(particles[:,3])
        cumulative_sum[-1] = 1.
        indices = np.searchsorted(cumulative_sum, np.random.random(n_particles))
        particles[:] = particles[indices]
        particles[:,3] = 1/n_particles
    return particles

def init_particles(n, x_range, y_range):
    """ 
    Generates n particles uniformly distributed over the x and y ranges 
    
    Each particle is a tuple [x, y, theta, w] where x is x position, y is y
    position, theta is yaw angle, and w is the particle's weight

    @param n: The number of particles to initialize
    @param x_range: A tuple (min_x, max_x)
    @param y_range: A tuple (min_y, max_y)
    """
    particles = np.empty((n, 5))
    particles[:,0] = np.random.uniform(x_range[0], x_range[1], size=n)  
    particles[:,1] = np.random.uniform(y_range[0], y_range[1], size=n)
    particles[:,2] = np.random.uniform(0, 2*np.pi, size=n)
    particles[:,3] = 1/n
    return particles

def particle_filter(marker_pos, marker_col, x_range=(0,200), y_range=(0,200), n_particles=1000, sensor_noise=3, save=False):
    """
    Simulation of particle filter algorithm

    @param marker_pos: A (n_markers, 2) numpy array where each entry i is the (x,y) position
    of the marker i

    @param marker_col: A (n_markers, 1) numpy array where each entry i is the color
    of the marker i

    @param x_range: A tuple (min_x, max_x)
    
    @param y_range: A tuple (min_y, max_y)
    
    @param n_particles: Number of particles to use

    @param sensor_noise: Standard deviation of gaussian sensor noise
    """
    if marker_pos.shape[0] != marker_col.shape[0]:
        return
    num_markers = marker_pos.shape[0]
    fig = plt.figure()

    # plot markers
    ax = plt.axes(xlim=x_range, ylim=y_range)
    ax.scatter(marker_pos[:,0], marker_pos[:,1], color=marker_col, s=300, marker='H', zorder=100, edgecolors='black', linewidth=2)

    # plot particles
    particles = init_particles(n=n_particles, x_range=x_range, y_range=y_range)
    particles_pos, = ax.plot(particles[:,0], particles[:,1], alpha=0.2, markersize=5, linestyle='None', marker='o',c='b', zorder=1)
    particles_dir = ax.quiver(particles[:,0], particles[:,1], np.cos(particles[:,2]), np.sin(particles[:,2]), alpha=0.2, color='b', scale=50, zorder=1)
    
    # plot robot pose
    inital_pos = np.array([0, y_range[1]/4])
    initial_dir = 0
    robot_pos, = ax.plot([inital_pos[0]], [inital_pos[1]], marker='o', markersize=10, c='black', zorder=1)
    robot_dir = ax.quiver([inital_pos[0]], [inital_pos[1]],np.cos(initial_dir),np.sin(initial_dir), color='black', scale=15, zorder=2)
    est_pos, =  ax.plot([0],[0], marker='x', markersize=10, c='red' )

    prev_pos = inital_pos
    prev_dir = initial_dir

    def animate(i):
        nonlocal prev_pos
        nonlocal prev_dir
        nonlocal particles

        # calculate robot's new direction and position
        command = np.array([1, 0])
        new_dir = prev_dir + command[1]
        new_pos = prev_pos + np.array([command[0]*np.cos(new_dir), command[0]*np.sin(new_dir)])

        # predicton step on all particles
        particles = prediction(particles, command)

        # calculate observations of markers, and reweight particles accordingly
        x_i = np.linalg.norm(new_pos-marker_pos, axis=1) + np.random.randn(num_markers)*sensor_noise/10
        particles = compute_weights(particles, marker_pos, x_i, sensor_noise )

        # resample particles according to new weights
        particles = resample(particles)

        # estimate pose
        mean = np.average(particles[:,:2], weights=particles[:,3], axis=0)
        est_pos.set_data([mean[0]], [mean[1]]) 

        # update robot and particle's position and direction in plot
        robot_pos.set_data([new_pos[0]], [new_pos[1]])
        robot_dir.set_offsets([new_pos[0], new_pos[1]])
        robot_dir.set_UVC([np.cos(new_dir)], [np.sin(new_dir)])
        particles_pos.set_data(particles[:,0], particles[:,1])
        particles_dir.set_offsets(np.array([particles[:,0], particles[:,1]]).T)
        particles_dir.set_UVC([np.cos(particles[:,2])], [np.sin(particles[:,2])])

        prev_pos = new_pos
        prev_dir = new_dir

        return particles_pos, particles_dir, robot_pos, robot_dir, est_pos

    # show animation
    anim = animation.FuncAnimation(fig, animate, frames=x_range[1], interval=20, blit=True)
    if (save):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('particle_filter.mp4', writer=writer)
    else:
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    np.random.seed(2) 
    particle_filter(np.array([[25,100], [75,100],[125,100],[175,100]]), 
                    np.array(['green', 'green', 'green', 'green']))

