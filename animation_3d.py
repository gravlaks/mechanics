import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.lines as lines
import numpy as np
from scipy.integrate import odeint
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
""" fig = plt.figure()
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


xMin = -10
xMax = 10
yMin = -10
yMax = 10
zMin = -2
zMax = 2
ax.set_xlim3d(xMin, xMax)
ax.set_ylim3d(yMin, yMax)
ax.set_zlim3d(zMin,zMax)
x = np.arange(xMin, xMax, 0.05)
y = np.arange(yMin, yMax, 0.05)
X, Y = np.meshgrid(x, y)
def fun(x,y):
    return 2*x-y
zs = np.array(fun(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
ax.plot_surface(X,Y,Z, alpha=0.3)

anim = animation.FuncAnimation(fig, animate,  
                     frames=N, interval=200, blit=False)
plt.show()
 """

def residual(du, u, p, t):
    g = 9.81
    m = 1

    x1 = u[0]
    y1 = u[1]
    z1 = u[2]
    x2 = u[3]
    y2 = u[4]
    z2 = u[5]
    x1_dot = u[6]
    y1_dot = u[7]
    z1_dot = u[8]
    x2_dot = u[9]
    y2_dot = u[10]
    z2_dot = u[11]
    lambda_1 = u[12]
    lambda_2 = u[13]
    lambda_3 = u[14]

    resid1 = du[0] - x1_dot
    resid2 = du[1] - y1_dot
    resid3 = du[2] - z1_dot
    resid4 = du[3] - x2_dot
    resid5 = du[4] - y2_dot
    resid6 = du[5] - z2_dot

    resid7 = lambda_1*x1 + lambda_3(x1-x2)
    resid8 = lambda_1*y1 + lambda_3(y1-y2)
    resid9 = -lambda_1- g*m + lambda_3(z1-z2)
    resid10 = lambda_2*x2 - lambda_3(x1-x2)
    resid11 = lambda_2*y2 - lambda_3(y1-y2)
    resid12 = -lambda_2-g*m - lambda_3(z1-z2)

    resid13 = (x1*du[6]+x1_dot**2) + (y1*du[7] + y1_dot**2)-du[8]
    resid14 = (x1*du[9]+x1_dot**2) + (y1*du[10] + y1_dot**2)-du[11]
    resid15 = (x1-x2)*(du[6]-du[9]) + (x1_dot-x2_dot)*(x1_dot-x2_dot)
    resid16 = (y1-y2)*(du[7]-du[10]) + (y1_dot-y2_dot)*(y1_dot-y2_dot)
    resid17 = (z1-z2)*(du[8]-du[11]) + (z1_dot-z2_dot)*(z1_dot-z2_dot)

    return np.array([resid1, resid2, resid3, resid4, resid5, resid6, resid7, resid8, resid9, resid10, resid11, resid12, resid13, resid14, resid15, resid16, resid17])

#initial conditions
t0 = 0.0
u0 = [1.0,1.0,1.0, 0., 0., 0., 0.,0.,0.,0.,0.,0., 0,0,0]
du0 = [0.,0.,0., 0., 0., 0., 0.,0.,0.,0.,0.,0., 0,0,0]

model = Implicit_Problem(residual, u0, du0, t0)
model.name = 'masses linked on surface'

sim = IDA(model)
tfinal = 10.0        #Specify the final time
ncp = 500            #Number of communication points (number of return points)

t, y, yd = sim.simulate(tfinal, ncp)
sim.plot()