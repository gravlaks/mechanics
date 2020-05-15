import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.lines as lines
import numpy as np
from scipy.integrate import odeint

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x0, y0, z0 = 1,1,1
def dynamics(q, t):
    y0 = 2
    g = 9.81
    m = 1
    k = 5
    d = 1



    theta = q[0]
    theta_dot = q[1]
    y = q[2]
    y_dot = q[3]

    theta_dot_dot = (-2*theta_dot*y_dot -g*np.sin(theta))/y
    y_dot_dot = (theta_dot**2)*y - (d/m)*y_dot + g*np.cos(theta) - (k/m)*(y-y0)
    dqdt = [theta_dot, theta_dot_dot, y_dot, y_dot_dot]
    return dqdt


def getPos(q):
    theta = q[0]
    theta_dot = q[1]
    y = q[2]
    y_dot = q[3]

    x_pos = y*np.sin(theta)
    y_pos = -y*np.cos(theta)
    z_pos = z0
    return (x_pos, y_pos, z_pos)

theta_init = np.pi/2
theta_dot_init = 0
y_init = 4
y_dot_init = 3

q_init = [theta_init, theta_dot_init, y_init, y_dot_init]

N = 100
tf = 20
t = np.linspace(0, tf,N)

init_pos = getPos(q_init)
state_traj = odeint(dynamics, q_init, t)
sc = ax.scatter((init_pos[0]), (init_pos[1]), (init_pos[2]), cmap='Greens')

def init():
    pass
    #return sc,
    
    #patch.center = (getPos(q_init))
    #ax.add_patch(patch)

    #return line, patch,   


def animate(i):
    q = state_traj[i]
    pos = getPos(q)

    sc._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
    print(sc._offsets3d)
    #return sc,

ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-10,10)
ax.set_zlim3d(-2,2)
anim = animation.FuncAnimation(fig, animate,  
                     frames=N, interval=200, blit=False)
plt.show()