import numpy as np

class fields():
  def __init__(self, E, B, mode='Analytical'):
    if mode == 'Analytical':
      self.E = E
      self.B = B
      self.mode = mode
    elif mode == 'Numerical':
      print('Not programmed module')
      self.mode = mode
      TypeError()
  def get_fields_analytical(self, x, t):
    E_mat = np.zeros(x.shape)
    B_mat = np.zeros(x.shape)
    for k in range(x.shape[0]):
        E_mat[k,:]=self.E(x[k,:],t)
        B_mat[k,:]=self.B(x[k,:],t)
    return E_mat,B_mat
  def get_fields_numerical():
    print('Not programmed module')
    TypeError()
