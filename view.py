from tkinter import *
from model import *
import time

root = Tk()
canvas = Canvas(root, bg='white', width='800', height='600')
canvas.pack()

def to_canvas_coords(canvas, x):
    h = canvas.winfo_reqheight()
    w = canvas.winfo_reqwidth()
    x = (h / 20) * x
    x.y = -x.y
    x.x += w/2
    x.y += h/2
    
    return x

def move_oval_to(canvas, o, u1, u2):
    u1 = to_canvas_coords(canvas, u1)
    u2 = to_canvas_coords(canvas, u2)

    canvas.coords(o, u1.x, u1.y, u2.x, u2.y)

def create_oval(canvas, particle):
    oval = canvas.create_oval(0, 0, 0, 0, fill='blue')
    move_oval_to(canvas, oval, particle.bounding_box()[0], particle.bounding_box()[1])

    return oval

def simulation_loop(constant_gravitational_field, timestep, particles):
    objects = []
    for particle in particles:
        object = create_oval(canvas, particle)
        objects.append(object)

    
    while True:
        constant_gravitational_field(timestep, particles, g=10)

        for oval, particle in zip(objects, particles):
            particle.inertial_move(timestep)
            move_oval_to(canvas, oval, particle.bounding_box()[0], particle.bounding_box()[1])
            
        canvas.update()
        time.sleep(timestep)
            
