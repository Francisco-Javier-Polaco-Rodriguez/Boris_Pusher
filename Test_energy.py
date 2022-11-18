from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from Particles_class import particles, particles_resembled
from Pusher_class import Boris_pusher
from scipy.constants import mu_0,epsilon_0,m_e,m_p,e,pi,c
from threading import Thread

B0 = 1

wc_e = e*B0/(m_e)
wc_p = e*B0/(m_p)

Ncores = 1
Npart_eachcore = 2
electrons = []
protons = []
for k in range(Ncores):
  x0, v0 =1e-6*np.random.rand(Npart_eachcore,3), (np.random.rand(Npart_eachcore,3)-0.5 * np.ones([Npart_eachcore,3]))*1e6
  electrons.append(particles(x0*1e-3, v0, m_e, -e))
for k in range(Ncores):
  x0, v0 =1e-6*np.random.rand(Npart_eachcore,3), (np.random.rand(Npart_eachcore,3)-0.5 * np.ones([Npart_eachcore,3]))*1e6
  protons.append(particles(x0, v0, m_p, e))



E = lambda x,t : np.array([0,0,0])
B = lambda x,t : np.array([0,0,B0])

dt_e = 0.05*wc_e**-1
dt_p = 0.05*wc_p**-1

test_electrons_pusher = []
for k in range(Ncores):
  test_electrons_pusher.append(Boris_pusher(electrons[k],dt_e,E,B,mode = 'Analytical'))
test_protons_pusher = []
for k in range(Ncores):
  test_protons_pusher.append(Boris_pusher(protons[k],dt_p,E,B,mode = 'Analytical'))

N= 1000000

print('\nSimulation of test electrons\n %i Threads\n %i Steps of time so %f cyclotrons periods time simulation\n %i Particles per thread\n'%(Ncores,N,N*dt_e*wc_e*2*pi,Npart_eachcore))
th_el = []
for k in range(Ncores):
  th_el.append(Thread(target = test_electrons_pusher[k].simulate_opt, args = (N,)))
for thread in th_el:
  thread.start()
for thread in th_el:
  thread.join()

print('\nSimulation of test protons\n %i Threads\n %i Steps of time so %f cyclotrons periods time simulation\n %i Particles per thread\n'%(Ncores,N,N*dt_p*wc_p*2*pi,Npart_eachcore))
th_pt = []
for k in range(Ncores):
  th_pt.append(Thread(target = test_protons_pusher[k].simulate_opt, args = (N,)))
for thread in th_pt:
  thread.start()
for thread in th_pt:
  thread.join()

electrons_ = particles_resembled(electrons)
protons_ = particles_resembled(protons)

electrons_.plot_trajectory()
protons_.plot_trajectory()
electrons_.plot_energy_in_time(wc_e**-1*2*pi,'2\pi\omega_{ce}^{-1}')
protons_.plot_energy_in_time(wc_p**-1*2*pi,'2\pi\omega_{cp}^{-1}')
plt.show()