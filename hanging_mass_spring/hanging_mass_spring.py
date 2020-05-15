import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.lines as lines
import numpy as np
from scipy.integrate import odeint

#rectangle = plt.Rectangle((10,10), 100, 100, fc='r', ec='y')
#plt.gca().add_patch(rectangle)


fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7,6.5)

ax = plt.axes(xlim=(-50,50), ylim=(-15, 20))
patch = plt.Circle((5, -5), 0.75,fc='b')

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
    return (x_pos, y_pos)

theta_init = np.pi
theta_dot_init = 2
y_init = 4
y_dot_init = 3
N = 100
tf = 20
t = np.linspace(0, tf,N)
q_init = [theta_init, theta_dot_init, y_init, y_dot_init]
init_pos = getPos(q_init)
state_traj = odeint(dynamics, q_init, t)
line, = plt.plot((0,init_pos[0]), (0, init_pos[1]), lw=2, color='black')

def init():
    patch.center = (getPos(q_init))
    ax.add_patch(patch)
    ax.add_line(line)

    return line, patch,   


def animate(i):
    q = state_traj[i]
    pos = getPos(q)

    line.set_xdata((0,pos[0]))
    line.set_ydata((0, pos[1]))
    patch.center = pos
    return line, patch,


anim = animation.FuncAnimation(fig, animate, init_func=init, 
                     frames=N, interval=200, blit=True)
plt.draw
plt.show()
