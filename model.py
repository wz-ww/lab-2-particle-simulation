import math

# Task (2/12): Define a class Vec

# Task (3/12): Additionally define a function dot(u, v)

# Task (4/12): Create a class Particle

# Task (5/12): In the Particle class, implement a method inertial_move(self, dt).

# Task (6/12): In the Particle class, implement a method apply_force(self, dt, f)

##########################################
### NB. Tasks 7–8 are done in view.py. ###
##########################################


# Task (9/12): In the Particle class, add a method bounding_box(self)






###########################################
### When you're done with all 12 tasks: ###
### forces/other features in this file! ###
###########################################

class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'
    
    def __rmul__(self, factor):
        return Vec(self.x * factor, self.y * factor)
    
    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)
    
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def get_coords(self):
        return (self.x, self.y)
    
def dot(u, v):
    return u.x * v.x + u.y * v.y

class Particle:
    def __init__(self, mass, position, velocity, radius):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.radius = radius
    
    def inertial_move(self, dt):
        self.position += self.velocity.__rmul__(dt)

    def apply_force(self, dt, f):
        self.velocity += f.__rmul__(dt)

    def bounding_box(self):
        a = Vec(self.position.x - self.radius, self.position.y + self.radius)
        b = Vec(self.position.x + self.radius, self.position.y - self.radius)
        
        return a, b

def constant_gravitational_field(dt, particles, g=10):
    for particle in particles:
        f = g * particle.mass * Vec(0, -1)
        particle.apply_force(dt, f)
        