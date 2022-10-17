import numpy as np
import matplotlib.pyplot as plt
from Particles_class import particles
from Pusher_class import Boris_pusher
from scipy.constants import mu_0,epsilon_0,m_e,m_p,e,pi,c


Npart = 2
x0, v0 =1e-6*np.random.rand(Npart,3), (np.random.rand(Npart,3)-0.5 * np.ones([Npart,3]))*1e3
electrons = particles(x0, v0, m_e, -e)
protons = particles(x0, v0, m_p, e)


E = lambda x,t : np.array([0,0,0])
B = lambda x,t : np.array([0,0,1])
dt = 1e-9
test_electrons_pusher = Boris_pusher(electrons,dt,E,B,mode = 'Analytical')
test_protons_pusher = Boris_pusher(protons,dt,E,B,mode = 'Analytical')

for k in range(5000):
  test_electrons_pusher.forward()
  test_protons_pusher.forward()


electrons.plot_trajectory()
plt.show()
protons.plot_trajectory()
plt.show()