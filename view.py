# Task (7/12): Draw on canvas

# Task (8/12): Define a new function to_canvas_coords(canvas, x)

#######################################
### NB. Task 9 is done in model.py. ###
#######################################

# Task (10/12): Define a new function move_oval_to(o, u1, u2)

# Task (11/12): Define a new function create_oval(canvas, particle)

# Task (12/12): Define a function simulation_loop(f, timestep, particles)

from tkinter import *
from model import *



root = Tk()
canvas = Canvas(root, bg='white', width='800', height='600')
canvas.pack()

o = canvas.create_oval(80, 30, 140, 150, fill="blue")

def to_canvas_coords(canvas, x):
    h = canvas.winfo_reqheight()
    w = canvas.winfo_reqwidth()
    x = x.__rmul__(h/20)
    x.y = -x.y
    x.x += w/2
    x.y += h/2
    
    return x

def move_oval_to(o, u1, u2):
    u1 = to_canvas_coords(canvas, u1)
    u2 = to_canvas_coords(canvas, u2)

    canvas.coords(o, u1.x, u2.x, u1.y, u2.y)

def create_oval(canvas, particle):
    pass




# input()