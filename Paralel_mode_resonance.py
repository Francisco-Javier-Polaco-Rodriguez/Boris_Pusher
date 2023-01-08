from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from Particles_class import particles, particles_resembled
from Pusher_class import Boris_pusher
from scipy.constants import mu_0,epsilon_0,m_e,m_p,e,pi,c
from threading import Thread

#Normalization to B0
B0 = [-1,0,0]
E0 = [0,0,0]

norm_B0 = np.linalg.norm(B0)
wc_e = e*norm_B0/(m_e)
wc_p = e*norm_B0/(m_p)

## Normalizing in function to Alfven velocity
V_A = 1

## Perturbation definition
w_wisteler = 5*wc_p
theta = 0
k0 = 2.044*wc_p/V_A
K_ = k0*np.array([np.cos(theta),0,np.sin(theta)])


B_wav = lambda x,t : np.linalg.norm(B0)*1e-4*np.array([0,np.cos(w_wisteler*t+np.dot(K_,x)),np.sin(w_wisteler*t+np.dot(K_,x))])
E_wav = lambda x,t : np.linalg.norm(B0)*1e-4*np.array([0,-np.sin(w_wisteler*t+np.dot(K_,x)),np.cos(w_wisteler*t+np.dot(K_,x))])/c

Ncores = 1
Npart_eachcore = 2000

electrons = []
protons = []
for k in range(Ncores):
  x0, v0 = 1e-6*np.random.rand(Npart_eachcore,3), 2*(np.random.rand(Npart_eachcore,3)-0.5 * np.ones([Npart_eachcore,3]))*6000
  electrons.append(particles(x0*1e-3, v0, m_e, -e))
for k in range(Ncores):
  x0, v0 = 1e-6*np.random.rand(Npart_eachcore,3), 2*(np.random.rand(Npart_eachcore,3)-0.5 * np.ones([Npart_eachcore,3]))*6000
  protons.append(particles(x0, v0, m_p, e))


E = lambda x,t : np.array(E0) + E_wav(x,t)
B = lambda x,t : np.array(B0) + B_wav(x,t)

dt_e = 0.1*wc_e**-1
dt_p = 0.1*wc_p**-1

test_electrons_pusher = []
for k in range(Ncores):
  test_electrons_pusher.append(Boris_pusher(electrons[k],dt_e,E,B,mode = 'Analytical'))
test_protons_pusher = []
for k in range(Ncores):
  test_protons_pusher.append(Boris_pusher(protons[k],dt_p,E,B,mode = 'Analytical'))

N= 2000

print('\nSimulation of test electrons\n %i Threads\n %i Steps of time so %1.0f cyclotrons periods time simulation\n %i Particles per thread (%i particles in total) \n'%(Ncores,N,N*dt_e*wc_e*2*pi,Npart_eachcore,Ncores*Npart_eachcore))
th_el = []
for k in range(Ncores):
  th_el.append(Thread(target = test_electrons_pusher[k].simulate_opt, args = (N,)))
for thread in th_el:
  thread.start()
for thread in th_el:
  thread.join()

#print('\nSimulation of test protons\n %i Threads\n %i Steps of time so %1.0f cyclotrons periods time simulation\n %i Particles per thread\n'%#(Ncores,N,N*dt_p*wc_p*2*pi,Npart_eachcore))
#th_pt = []
#for k in range(Ncores):
#  th_pt.append(Thread(target = test_protons_pusher[k].simulate_opt, args = (N,)))
#for thread in th_pt:
#  thread.start()
#for thread in th_pt:
#  thread.join()

electrons_ = particles_resembled(electrons)
#protons_ = particles_resembled(protons)


#electrons_.plot_trajectory()
#protons_.plot_trajectory()
electrons_.plot_dmu2_Vx()
#protons_.plot_dmu2_Vx()
plt.show()

electrons_.save('Wisteler_wave_theta=%1.2f_%i_time_steps.mat'%(theta,N))