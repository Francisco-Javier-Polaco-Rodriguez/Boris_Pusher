import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0,epsilon_0,m_e,m_p,e,pi,c
from scipy.io import savemat
class particles():
  def __init__(self, x0, v0, m,q):
    # Axis notation 0 = particles, 1 = space x y z, 2 = time
    self.x = x0[:,:,np.newaxis]
    self.v = v0[:,:,np.newaxis]
    self.m = m
    self.q = q
    self.firstx = True
    self.firstv = True
    self.dt = []

  def x_forward(self, dt):
    # It computes the position of each particle.
    if self.firstx:
      self.firstx = False
      self.x = np.concatenate(
        (self.x,
         self.x[:, :, -1][:, :, np.newaxis] + self.v[:, :, -1][:, :, np.newaxis] * dt * 0.5),
        axis=2)
      self.dt = dt
    else:
        self.x = np.concatenate((self.x, self.x[:, :, -1][:, :, np.newaxis] +
                                 self.v[:, :, -1][:, :, np.newaxis] * self.dt),
                                axis=2)
  def v_forward(self, v_new):
    self.v = np.concatenate((self.v, v_new[:, :, np.newaxis]), axis=2)

  def get_last_step(self):
    return self.m,self.q,self.x[:,:,-1],self.v[:,:,-1]

  def get_mean_energy_in_time(self):
    return 0.5*self.m*np.mean(np.linalg.norm(self.v,axis = 1)**2,axis = 0)/e

  def t(self):
    return np.arange(0,self.x.shape[2],1)*self.dt

  def plot_trajectory(self,colord = 'b'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for k in range(self.x.shape[0]):
        ax.plot3D(self.x[k,0,:],self.x[k,1,:],self.x[k,2,:],colord,alpha = 0.5)
    ax.set_xlabel('x    (m)',size = 13)
    ax.set_ylabel('y    (m)',size = 13)
    ax.set_zlabel('z    (m)',size = 13)
    ax.set_title('Particles trajectories')
    return fig
  def mean_Vx(self):
    return np.sum(self.v[:,0,:],axis = 1)/self.t().shape[0]
  def mu(self):
    return np.sum(self.v[:,0,:] / np.linalg.norm(self.v,axis = 1),axis = 1)/self.t().shape[0]
  def dmu2(self):
    return np.sum((self.v[:,0,:] / np.linalg.norm(self.v,axis = 1)-self.mu()[:,np.newaxis])**2 , axis = 1)/self.t().shape[0]
  def plot_dmu2_Vx(self):
    fig = plt.figure()
    plt.plot(self.mean_Vx(),self.dmu2(),'.r')
    plt.title(r'$\langle\delta\mu\delta\mu\rangle(\langle V_{e\parallel}\rangle)$')
    plt.yscale('log')
    plt.ylabel(r'$\langle \delta\mu\delta\mu\rangle/V_A^2$ ',size = 13)
    plt.xlabel(r'$\langle V_{e\parallel}\rangle/V_A$')
    plt.grid()
  def plot_energy_in_time(self,timescale,name_timescale,colord = 'b'):
    fig = plt.figure()
    plt.plot(self.t()/timescale,self.get_mean_energy_in_time())
    plt.title('Energy in time')
    plt.ylabel(r'$\langle E\rangle$  (eV)',size = 13)
    plt.xlabel(r't   ($%s$)'%(name_timescale))
    plt.grid()
  def save(self,name):
    savemat(name,{'v':self.v,'x':self.x,'t':self.t})


class particles_resembled(particles):
  def __init__(self,list_of_particles_classes):
    for i,particles in enumerate(list_of_particles_classes):
      if i == 0:
        self.v = particles.v
        self.x = particles.x
        size = self.x.shape[2]
        self.q = particles.q
        self.m = particles.m
        self.dt = particles.dt
      else:
        if particles.q != self.q or particles.m != self.m or self.dt != particles.dt:
          TypeError('Particles has not the same q m dt properties.')
        else:
          self.v = np.concatenate((self.v,particles.v),axis = 0)
          self.x = np.concatenate((self.x,particles.x),axis = 0)
