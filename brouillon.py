#trying some shit out there

import numpy as np
from matplotlib import pyplot as plt

class Mouton():
  positiion = np.array(2, dtype = float) #coords cartésiennes
  vitesse = np.array(2, dtype = float) #coords polaires
  
  def __init__(self):
    position = np.random.uniform(2)
    vitesse = np.array([1, np.random.uniform()*2*np.pi])
  
  def update(self):
    vit_cart = np.array([vitesse[0] * np.cos(vitesse[1]), vitesse[0] * np.sin(vitesse[1])])
    position += vit_cart

class Model(n, g_func = None):
  A = np.array((n,n))
  g = function
  G = np.array((n,n))
  eps = function
  
  def __init__(self):
    A = np.zeros((n, n))
    for i in range(n-1):
      A[i][i+1] = 1
    g = lambda x1, t1, x2, t2 : sin( np.angle((x1 - x2)[0] + (x1 - x2)[1]j) - t1)
    
    if g_func != None:
      g = g_func
    
    eps = lambda : np.random.uniform()
  
  
class Simulation(N, affichage = False, limites = 'BVK'):
  Population = np.array(N, dtype = Mouton)
  modele = Model(N)
  
  def evolve(self):
    True
  
  
  
