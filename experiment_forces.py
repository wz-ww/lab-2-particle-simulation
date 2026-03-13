from view import *
from model import *
import math

# tests gravitational field with constant downward force, we see the particles moving downward, which is the expected behaviour, 
# and the speed of this increases as we increase the size of the downward vector
# timestep is frames per second, so decreasing it makes the particles update less often and vice versa

n = 20
particles = []
for i in range(n):
    theta = i*2*math.pi/n
    u = Vec(math.cos(theta),math.sin(theta))
    pos = 10 * u
    vel = -1 * u 
    particles.append(Particle(1,pos,vel,0.2))

def no_force(dt,particles):
    pass

f = constant_gravitational_field

simulation_loop(f, 0.005, particles)
