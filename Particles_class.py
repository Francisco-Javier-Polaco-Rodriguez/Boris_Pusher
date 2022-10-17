import numpy as np
import matplotlib.pyplot as plt
class particles():

  def __init__(self, x0, v0, m,q):
    # Axis notation 0 = particles, 1 = space x y z, 3 = time
    self.x = x0[:,:,np.newaxis]
    self.v = v0[:,:,np.newaxis]
    self.m = m
    self.q = q
    self.firstx = True
    self.firstv = True

  def x_forward(self, dt):
    # It computes the position of each particle.
    if self.firstx:
      self.firstx = False
      self.x = np.concatenate(
        (self.x,
         self.x[:, :, -1][:, :, np.newaxis] + self.v[:, :, -1][:, :, np.newaxis] * dt * 0.5),
        axis=2)
    else:
        self.x = np.concatenate((self.x, self.x[:, :, -1][:, :, np.newaxis] +
                                 self.v[:, :, -1][:, :, np.newaxis] * dt),
                                axis=2)
  def v_forward(self, v_new):
    self.v = np.concatenate((self.v, v_new[:, :, np.newaxis]), axis=2)
    

  def get_last_step(self):
    return self.m,self.q,self.x[:,:,-1],self.v[:,:,-1]

  def plot_trajectory(self,colord = 'b'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for k in range(self.x.shape[0]):
        ax.plot3D(self.x[k,0,:],self.x[k,1,:],self.x[k,2,:],colord,alpha = 0.5)
    ax.set_xlabel('x',size = 13)
    ax.set_ylabel('y',size = 13)
    ax.set_zlabel('z',size = 13)
    ax.set_title('Particles trajectories')
    return fig