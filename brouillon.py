#trying some shit out there

import numpy as np
from matplotlib import pyplot as plt

def cartesianND(t) : #t un tableau nD en coordonnées polaires sur la dernière dimension (n1, n2..., n(d-1), 2)
  return np.array([t.T[0]*np.cos(t.T[1]), t.T[0]*np.sin(t.T[1])]).T
  


class Mouton():
  position = np.empty((2,), dtype = float) #coords cartésiennes
  vitesse = np.empty((2,), dtype = float) #coords polaires
  
  def __init__(self):
    self.position = np.random.uniform(size = 2)
    self.vitesse = np.array([1, np.random.uniform()*2*np.pi])
  
  def get_cart_spd(self): #calcule la vitesse en coordonnées cartésiennes
    return np.array([self.vitesse[0] * np.cos(self.vitesse[1]),
                     self.vitesse[0] * np.sin(self.vitesse[1])])

  def update(self, thetap = 0):
    self.vitesse[1] += thetap
    self.position += self.get_cart_spd()
  


#  def __getitem__(self, items):
#    if len(items) == 2:
#      if items[0] == 0:
#        return self.position[items[1]]
#      elif items[0] == 1:
#        return self.vitesse[items[1]]
#    elif len(items) == 1:
#      if items[0] == 0:
#        return self.position
#      elif items[0] == 1:
#        return self.vitesse
#    else:
#      print(f'Error in Mouton.__getitem__ : item of size {len(items)}')
      
      
class Model():
  n = 2
  A = np.zeros((n,n))
  g = lambda x : x
  eps = lambda x : x
  D = 1
  
  def __init__(self, n0, bruit = 1):
    self.n = n0
    self.A = np.zeros((self.n, self.n))
    for i in range(self.n-1):
      self.A[i][i+1] = 1
      
    # g (mouton1, mouton2)
    self.g = lambda m1, m2 : np.sin( np.angle((m1[0] - m2[0])[0] + (m1[0] - m2[0])[1]*1j) - m1[1,1])
    self.D = bruit
    self.eps = lambda size = None : np.random.uniform(size = size)

  def G(self, M): # M matrice 3D de population de moutons (shape N, 2, 2)
    calc = np.array([[M[i,0] - M[j,0] for j in range(M.shape[0])] for i in range(M.shape[0])]) #SI PEU EFFICACE ! A AMELIORER
    alpha = np.angle(calc[:,:,0] + calc[:,:,1] * 1j)
    return np.sin(alpha - M[:,1,1])

class Simulation():
  N = 2
  Population = np.empty((N, 2, 2))
  modele = Model(N)
  verbose = False
  affichage = False
  limite = None
  def __init__(self, N = 2, verbose = False, affichage = False, limites = 'BVK'):
    self.verbose = verbose
    self.affichage = affichage
    self.limite = 'BVK'
    self.N = N
    self.modele = Model(self.N)
    self.Population = np.random.uniform(size = (N, 2, 2))
    self.Population[:,1,0] = 1
    self.Population[:,1,1] *= 2*np.pi

      
    if self.verbose :
      print(f'Initialisation de la simulation avec {N} moutons et des limites de type {limites}')
  
  def evolve(self):
    #stock_pop = self.Population.copy()
    
    self.Population[:,1,1] += (self.modele.A * self.modele.G(self.Population)).sum(axis = 0
                   ) + 2*self.modele.D*self.modele.eps(size = (self.N))
    
    self.Population[:,0] += cartesianND(self.Population[:,1])
    
    if self.limite == 'BVK':
      self.Population[:,0] %= 10
    

  def affiche_population(self):
    fig = plt.figure(figsize = (10, 10))
    plt.quiver(self.Population[:,0,0], self.Population[:,0,1], self.Population[:,1,0], self.Population[:,1,1])
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.hlines([-10, 10], -10, 10, ls = '--')
    plt.vlines([-10, 10], -10, 10, ls = '--')
    plt.show()
  
  def Simulate(self, n_step):
    
    for k in range(n_step):
      self.evolve()
      print(f'Evolved step {k}')
    print(f'Evolved for {n_step} steps')
    if self.affichage:
      self.affiche_population()
  
  
#Test zone 
    
    
x = Mouton()
#stock_x = x.position.copy()
#x.update()
#plt.subplot(121)
#plt.scatter(x.position[0], x.position[1])
#plt.scatter(stock_x[0], stock_x[1])
#plt.subplot(122, projection = 'polar')
#plt.scatter(x.vitesse[0], x.vitesse[1])
#plt.show()

y = Model(2)

z = Simulation(2, verbose = True)
