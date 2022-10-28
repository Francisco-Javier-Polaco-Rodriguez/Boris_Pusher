import numpy as np
from Fields_class import fields
from tqdm import tqdm

class Boris_pusher():

  def __init__(self, particles, dt, E, B, mode='Analytical'):
    self.particles = particles
    self.dt = dt
    self.fields = fields(E, B, mode=mode)
    self.k = 0

  def forward(self):
    dt = self.dt
    t = self.k*dt
    self.particles.x_forward(dt)
    m, q, x, v = self.particles.get_last_step()
    E, B = self.fields.get_fields_analytical(x, t)
    v_minus = v + q / m * E * dt / 2
    t = q * dt / 2 / m * B
    v_prime = v + np.cross(v_minus, t,axis = 1)
    s = t * 2 / (1 + np.linalg.norm(t, axis = 0)**2)
    v_plus = v_minus + np.cross(v_prime, s,axis = 1)
    v_new = v_plus + q * E / m * dt / 2
    self.particles.v_forward(v_new)
    self.k += 1
  def simulate(self,N):
    for k in tqdm(range(N)):
      self.forward()