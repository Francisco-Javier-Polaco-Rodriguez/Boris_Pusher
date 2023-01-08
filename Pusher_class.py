import numpy as np
from Fields_class import fields
from tqdm import tqdm

class Boris_pusher():

  def __init__(self, particles, dt, E, B, mode = 'Analytical'):
    self.particles = particles
    self.dt = dt
    self.fields = fields(E, B, mode=mode)
    self.k = 0

  def forward(self):
    dt = self.dt
    t_ = self.k*dt
    self.particles.x_forward(dt)
    m, q, x, v = self.particles.get_last_step()
    E, B = self.fields.get_fields_analytical(x, t_)
    v_minus = v + q / m * E * dt / 2
    t = q * dt / 2 / m * B
    v_prime = v_minus + np.cross(v_minus, t,axis = 1)
    s = t * 2 / (1 + np.linalg.norm(t, axis = 0)**2)
    v_plus = v_minus + np.cross(v_prime, s,axis = 1)
    v_new = v_plus + q * E / m * dt / 2
    self.particles.v_forward(v_new)
    self.k += 1

  def simulate(self,N):
    for k in tqdm(range(N)):
      self.forward()

  def forward_opt(self,xold,vold,t):
    dt = self.dt
    m = self.particles.m
    q = self.particles.q
    E, B = self.fields.get_fields_analytical(xold, t)
    v_minus = vold + q / m * E * dt / 2
    t_ = q * dt / 2 / m * B
    v_prime = v_minus + np.cross(v_minus, t_,axis = 1)
    s = t_ * 2 / (1 + np.linalg.norm(t_, axis = 1)[:,np.newaxis]**2)
    v_plus = v_minus + np.cross(v_prime, s,axis = 1)
    v_new = v_plus + q * E / m * dt / 2
    return v_new

  def simulate_opt(self,N):
    if self.particles.firstx:
      TypeError('Simulation already started with no optimization. If you want to use this mode you need to do the first step with simulate_opt.')
    self.particles.firstx,self.particles.firstv = False,False
    xold = self.particles.x[:,:,0]
    vold = self.particles.v[:,:,0]
    x = np.zeros([xold.shape[0],xold.shape[1],N])
    v = np.zeros([vold.shape[0],vold.shape[1],N])
    x[:,:,0],v[:,:,0] = xold,vold
    x[:,:,1]=xold+vold*self.dt*0.5
    v[:,:,1] = self.forward_opt(xold,vold,0)
    del xold,vold
    for k in tqdm(range(1,N-1)):
      x[:,:,k+1] = x[:,:,k]+v[:,:,k]*self.dt
      v[:,:,k+1] = self.forward_opt(x[:,:,k],v[:,:,k],k*self.dt)
    self.particles.x = x
    self.particles.v = v
    self.particles.dt = self.dt