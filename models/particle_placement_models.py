# NOTES:
# Sketch of code that could be used to randomly place particles inside of a given shape.
# This would avoid the need for a grid of particle / pixels to be used.
# Then, a circle hole patch could be applied to the particle positions instead of applied to a particle grid.

#%%
import numpy as np

def rect_filter(x, x_nought=[0,0], width=1.0, height=1.0):
    x_prime = x - x_nought
    return x_prime[0] > 0 and x_prime[0] < width and x_prime[1] > 0 and x_prime[1] < height

def circle_filter(x, x_nought=[0.5,0.5], radius=0.4):
    x_prime = x - x_nought
    return np.power(x_prime, 2).sum() < radius **2

def right_side_up_triangle(x, x_nought=[0,0], base=1, height=1):
    x_prime = x - x_nought
    above_ground = lambda x: x[0] > 0 and x[1] > 0
    right_of_left_edge = lambda x: True
    left_of_right_edge = lambda x: True

    return above_ground(x_prime) and right_of_left_edge(x_prime) and left_of_right_edge(x_prime)



class particle_placement_model():
    def __init__(self,
                    particle_count = 1024,
                    placement_filter=lambda x: True):
        self.particle_count = particle_count
        self.placement_filter = placement_filter

        self.particles = []
        self.place_particles()

    def get_positions(self):
        return self.particles

    def place_particles(self):
        while len(self.particles) < self.particle_count:
            x = np.random.random(2) * 2 -1 
            if self.placement_filter(x):
                self.particles.append(x)
#%%
import matplotlib.pyplot as plt

def plot(placement_filter):
    ppm = particle_placement_model(placement_filter=placement_filter)
    x,y = zip(*ppm.get_positions())
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    ax.scatter(x,y)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    plt.show()

#%%
def dot (a,b):
    return np.dot(a, b)

def clamp(x, min_val, max_val):
    return min(max(x, min_val), max_val)

def sign (x):
    return -1 if x < 0 else 1

def length(x):
    return np.linalg.norm(x)

def star(x, r=0.5, rf=.50):
    k1 = np.array([0.809016994375, -0.587785252292])
    k2 = np.array([-k1[0],k1[1]])

    p = np.array(x)
    p[0] = abs(p[0])
    
    p -= 2.0*max(dot(k1,p),0.0)*k1
    p -= 2.0*max(dot(k2,p),0.0)*k2

    p[0] = abs(p[0])
    p[1] -= r
    ba = rf * np.array([-k1[1], k1[0]]) - np.array([0,1])
    h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r)
    sdf_val = length(p-ba*h) * sign(p[1]*ba[0]-p[0]*ba[1])
    # print(sdf_val)
    return sdf_val < 0.0

plot(star)

#%%
for placement_filter in [lambda x: True, rect_filter, circle_filter]:
    plot(placement_filter)
# %%


mask_from_img_url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Platypus_illustration.jpg"
from PIL import Image
import requests
from io import BytesIO

response = requests.get(mask_from_img_url)
img = Image.open(BytesIO(response.content)).resize((64,64))
img_mask = (np.array(img)[:,:,0] < 255)
# %%
# %%
