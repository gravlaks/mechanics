import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.lines as lines
import numpy as np
from scipy.integrate import odeint

p0 = [0,0,0]
p_Nplus1 = [5, 0, 0]
def dynamics(state, t):
    g = 9.81
    m = 1
    K = 2
    N = (len(state)//(3*2))-2
    print(N)
    q = []
    q_dot = []
    q_dot_dot =[]
    

    for i in range((N+2)*3):
        q_dot   += [state[i + (N+2)*3]]
        q       += [state[i]]
    q_dot_dot  += [0.,0.,0.]
    # Pulling the rope
    force = 50
    if (abs(t-7) <0.1):
        q_dot_dot[2] =force
    if (abs(t-7.4) <0.1):
        q_dot_dot[2] =-force
    if (abs(t-7.6) <0.1):
        q_dot_dot[2] =-force
    if (abs(t-8.0) <0.1):
        q_dot_dot[2] =force
    if (abs(t-8.2) <0.1):
        q_dot_dot[2] =-force
    if (abs(t-8.6) <0.1):
        q_dot_dot[2] =force
    if (abs(t-8.8) <0.1):
        q_dot_dot[2] =force
    if (abs(t-9.2) <0.1):
        q_dot_dot[2] =-force



    for i in range(0,N):
        pi = q[i*3:i*3+3]
        p_ip1 = q[i*3+3:i*3+6]
        p_ip2 = q[i*3+6:i*3+9]
        print(pi, p_ip1, p_ip2)
        
        for idx in range(3):
            d= 0
            if q_dot[i*3+2] > 0 and (idx+1) % 3 == 0:
                d = 1
            elif q_dot[i*3+2] > 0 and (idx+1) % 3 == 0:
                d = -1
            if (idx+1) % 3 == 0:
                g = 9.81
            else:
                g = 0
            q_dot_dot += [+K/m*((pi[idx]-p_ip1[idx]) + (p_ip2[idx]-p_ip1[idx])) - g -d]
    q_dot_dot += [0.,0.,0.]
 

    state_dot = q_dot + q_dot_dot
    return state_dot
def getPos(state):
    N = (len(state)//(3*2))-2
    
    p = [0]*(N+2)
    p[0] = p0
    p[N+1] = p_Nplus1 
    for i in range(N+1):
        p[i] = [state[i*3], state[i*3+1], state[i*3+2]]
    return p



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xMin = -10
xMax = 10
yMin = -10
yMax = 10
zMin = -20
zMax = 20
ax.set_xlim3d(xMin, xMax)
ax.set_ylim3d(yMin, yMax)
ax.set_zlim3d(zMin,zMax)


state_init = [0.,0.,0, 1.,1.,-1., 2.,1.,-1., 2.5, 2.,-2, 5.,0.,0., 0.,0.,0., 0.,0.,0, 0.,0.,0, 0.,0.,0, 0.,0.,0]
print("state_init_length:", len(state_init))
init_pos = getPos(state_init)
t0 = 0.0
tf = 20
timesteps = 100
t = np.linspace(t0, tf,timesteps)
state_traj = odeint(dynamics, state_init, t)
N = (len(state_traj[0])//(3*2))-2
print("N: ", N)
print("length of trajectory: ", len(state_traj))
print("t: ", t)
init_xs = [init_pos[i][0] for i in range(0, N+2)]
init_ys = [init_pos[i][1] for i in range(0, N+2)]
init_zs = [init_pos[i][2] for i in range(0, N+2)]
sc = ax.scatter(init_xs, init_ys, init_zs, cmap='Greens')
lines = []


line, = plt.plot(init_xs, init_ys, init_zs, lw=1.5, color='black')
   

def init():
    ax.add_line(line)


def animate(i):
    print(i)
    state = state_traj[i]
    N = (len(state)//(3*2))-2

    positions = getPos(state)
    xs = [positions[idx][0] for idx in range(0, N+2)]
    ys = [positions[idx][1] for idx in range(0, N+2)]
    zs = [positions[idx][2] for idx in range(0, N+2)]

    sc._offsets3d = (xs, ys, zs)
    for i in range(N+1):
        line.set_data(xs,ys)
        line.set_3d_properties(zs)


    return sc, lines,
anim = animation.FuncAnimation(fig, animate,  
                     frames=timesteps, init_func=init, interval=200, blit=False)
plt.show()
