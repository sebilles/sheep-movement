#trying some shit out there

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as m_ani
import matplotlib as mplt
import time

rng = np.random.default_rng()


color_list = list(mplt.colors.TABLEAU_COLORS)

def cartesianND(t) : #t un tableau nD en coordonnées polaires sur la dernière dimension (n1, n2..., n(d-1), 2)
  return np.array([t.T[0]*np.cos(t.T[1]), t.T[0]*np.sin(t.T[1])]).T
  
def cartesian1D(t) : 
  return np.array([t[0]*np.cos(t[1]), t[0]*np.sin(t[1])])
  
def frame_producer(Pop, fig, ax, box_size):  
    
    ax.clear()
    im_list = []
    
    cart_spd= cartesianND(Pop[:,1])

    if len(Pop) <= len(color_list):

        im_list.append(ax.quiver(Pop[:,0,0], Pop[:,0,1], cart_spd[:,0], cart_spd[:,1],
                          width = 0.005, scale = 25, color = color_list[:len(Pop)]))
        
    else :
        im_list.append(ax.quiver(Pop[:,0,0], Pop[:,0,1], cart_spd[:,0], cart_spd[:,1],
                          width = 0.005, scale = 25))

    im_list.append(ax.hlines([0, box_size], 0, box_size, ls = '--'))
    im_list.append(ax.vlines([0, box_size], 0, box_size, ls = '--'))
    
    return im_list

  
    
  
  
class Model():
  n = 2
  A = np.zeros((n,n))
  g = lambda x : x
  eps = lambda x : x
  D = 1
  
  def __init__(self, n0, bruit = 0.01):
    self.n = n0
    self.A = np.zeros((self.n, self.n))
    for i in range(self.n-1):
      self.A[i][i+1] = 1
      
    # g (mouton1, mouton2)
    self.g = lambda m1, m2 : np.sin( np.angle((m1[0] - m2[0])[0] + (m1[0] - m2[0])[1]*1j) - m1[1,1])
    self.D = bruit
    self.eps = lambda size = None : rng.random(size = size) - 0.5


  def G(self, M, L = 1000): # M matrice 3D de population de moutons (shape N, 2, 2)
  
    calc = np.array([[M[i,0] - M[j,0] for j in range(M.shape[0])] for i in range(M.shape[0])]) #SI PEU EFFICACE ! A AMELIORER
    calc_bis = calc[:,:,0] + calc[:,:,1] * 1j
    
    #bruit sur les positions
    theta_bruit = rng.random(self.n) * 2 * np.pi
    r_bruit = rng.normal(0, (np.absolute(calc_bis)/L)**2)
    z_bruit = r_bruit * np.exp(1j * theta_bruit)
    
    alpha = np.angle(calc_bis+ z_bruit ) 
    
    return np.sin(alpha - M[:,1,1])
  
  def quantify(self, M, func = 'files'): #WIP
      if func == 'files':
        calc = np.array([[M[i,1, 1] - M[j,1, 1] for j in range(M.shape[0])] for i in range(M.shape[0])]) #SI PEU EFFICACE ! A AMELIORER
        return (self.A * np.cos(calc)).sum(axis = 1) / np.where(self.A.sum(axis = 1)!=0,
                                                                self.A.sum(axis = 1), 1)
      elif func == 'clusters':
        grav = np.mean(M[:,0], axis = 0)
        return np.abs((M[:,0] - grav)**2).sum()/self.n
    
A_stocker = []
    
class Simulation():
  N = 2
  Population = np.empty((N, 2, 2))   #Troupeau de moutons de forme [mouton1, mouton2...] avec mouton = [position, vitesse] avec position cartésienne et vitesse polaire (2D)
  modele = Model(N)
  verbose = False
  affichage = False
  limite = None
  box_size = 10    # Univers limité par un carré de côté box_size avec un sommet en (0,0)
  def __init__(self, N = 2, verbose = False, affichage = False, limites = 'BVK',
               dispersion = 10, box_size = 100, bruit = 0.01, vision = 100):   #Pour le moment bruit max à 1 et min à 0
    self.verbose = verbose
    self.affichage = affichage
    self.limite = limites 
    self.N = N
    self.L= vision #distance caractéristique de vision du mouton
    self.modele = Model(self.N, bruit = bruit)
    self.Population = rng.random(size = (N, 2, 2))
    self.box_size = box_size
    self.Population[:,0] = (self.Population[:,0] - 0.5)*dispersion + self.box_size/2
    self.Population[:,1,0] = 1
    self.Population[:,1,1] *= 2*np.pi

      
    if self.verbose :
      print(f'Initialisation de la simulation avec {N} moutons et des limites de type {limites}')
  
  def evolve(self, evolve_A = False):
    #stock_pop = self.Population.copy()
    
    if evolve_A:
        r = 2
        delta = 0.0001
        C = self.Population[:,0, 0] + self.Population[:,0,1]*1j
        xx, yy = np.meshgrid(C, C)
        R = np.absolute(xx - yy)
        T = np.angle(xx-yy)
        L = ((R > 0) & (R < r)) & ((T  < (np.pi / 4)) & (T > (-np.pi / 4)))
        self.modele.A += L * delta
        self.modele.A = np.round(self.modele.A  / np.where(self.modele.A.sum(axis = 1)!=0,
                                                  self.modele.A.sum(axis = 1), 1), 5)
        A_stocker.append(self.modele.A)
    
    
    self.Population[:,1,1] += (self.modele.A * self.modele.G(self.Population, 
                                                             L = self.L)).sum(axis = 0
                   ) + 2*np.pi*self.modele.D*self.modele.eps(size = (self.N))  #bruit max à 1, amplifié quand les autres sont loin ? mvt limité à pi/2?

    self.Population[:,1,1] %= 2*np.pi

    self.Population[:,0] += cartesianND(self.Population[:,1])
    
    if self.limite == 'BVK':
      self.Population[:,0] %= 100  #limite : boîte de taille 100x100
    
    if self.limite == 'Align':      #WIP, almost working
      collision_problems = np.count_nonzero((self.Population[:,0] < 0) + (self.Population[:,0] > self.box_size), axis = 1)
      
      if np.any(collision_problems):

      
          bool_pi = self.Population[:,1, 1] > np.pi
          bool_3pi_2 = self.Population[:,1, 1] > (3*np.pi/2)
          bool_pi_2 = self.Population[:,1, 1] < np.pi/2
          
            
          bool_x_0 = self.Population[:,0,0] < 0
          bool_x_b = self.Population[:,0,0] > self.box_size
          bool_y_0 = self.Population[:,0,1] < 0
          bool_y_b = self.Population[:,0,1] > self.box_size
         
          bool_x_0p = self.Population[:,0,0] > self.Population[:,1,0]
          bool_y_0p = self.Population[:,0,1] > self.Population[:,1,0]
          bool_x_bp = self.Population[:,0,0] < (self.box_size - self.Population[:,1,0])
          bool_y_bp = self.Population[:,0,1] < (self.box_size - self.Population[:,1,0])
        
    
          theta_prime = (np.pi*(  (bool_x_0)*((bool_y_0p & np.logical_not(bool_y_bp & bool_pi))                 + 0.5)
                                + (bool_x_b)*((bool_y_0p & np.logical_not(bool_y_bp & np.logical_not(bool_pi))) + 0.5)
                                + (bool_y_0)*((bool_x_0p & np.logical_not(bool_x_bp & (bool_3pi_2)))                 )
                                + (bool_y_b)*((bool_x_0p & np.logical_not(bool_x_bp & (bool_pi_2)))                  )
                                                                      
                                ) - self.Population[:,1, 1] )* collision_problems
        
      else:
          theta_prime = 0
        
      self.Population[:,0] -= cartesianND(self.Population[:,1])
      
      self.Population[:,1, 1] += theta_prime 
      
      self.Population[:,0] += cartesianND(self.Population[:,1])

      

          
      if self.verbose :
        print(f'Found {np.count_nonzero(collision_problems)} collisions')
      

  def affiche_population(self, for_ani = False): # for_ani = Artist production for animation
      
    cart_spd= cartesianND(self.Population[:,1])
    fig = plt.figure(figsize = (10, 10))
    
    
    plt.quiver(self.Population[:,0,0], self.Population[:,0,1], cart_spd[:,0], cart_spd[:,1],
               width = 0.005, scale = 25, color = color_list[:self.N])
    plt.hlines([0, self.box_size], 0, self.box_size, ls = '--')
    plt.vlines([0, self.box_size], 0, self.box_size, ls = '--')
    if not for_ani:
        plt.show()
    else:
        return [fig.gca()]
  def Simulate(self, n_step, animate = False, anim_name = 'Sheep_Sim_Test', evolve_A = False):
      
    if self.verbose:
      time_0 = time.time()
    
    alignement = np.empty((n_step+1, self.N), dtype = float)
    
    alignement[0] = self.modele.quantify(self.Population)
    
    if self.affichage and animate :
        figure, ax = plt.subplots(figsize = (10,10))
        Pops = np.empty((n_step+1, self.N, 2, 2))
        Pops[0] = self.Population.copy()

    for k in range(n_step):
      if self.affichage and animate:
        Pops[k+1] = self.Population.copy()
      elif self.affichage:
        self.affiche_population()
      self.evolve(evolve_A)
      alignement[k+1] = self.modele.quantify(self.Population)
      if self.verbose : 
          print(f'Evolved step {k}')
      
    print(f'Evolved for {n_step} steps')
    if self.verbose:
      time_1 = time.time()
    
    if self.affichage and animate:# + gaussienne en fonction de la distance
      ani = m_ani.FuncAnimation(figure, frame_producer, frames = Pops, fargs = (figure, ax, self.box_size), blit = False)
      # + gaussienne en fonction de la distance
      ani.save('Videos_Sheep/' + anim_name + '.mp4')
    elif self.affichage:
      self.affiche_population()
      
    if self.verbose:
      time_2 = time.time()
      
      print(f'Simulation time : {round(time_1 - time_0, 3)}s; animation time : {round(time_2 - time_1, 3)}s')
      
    return alignement




class Simulation_Pool():
    
    def __init__(self, N_sim = 3, N = 4, limites = 'Align', 
                 dispersion = 10, bruit = 0.01, vision = 100):
        self.N_sim = N_sim
        self.N = N
        self.limite = limites 
        self.L= vision 
        self.dispersion = dispersion
        self.bruit = bruit
        self.Sims_created = False
        self.simulated = False
    def stat_pool(self):
        self.Sims = [Simulation(N = self.N, limites = self.limite,
                     dispersion = self.dispersion, bruit = self.bruit,
                     vision = self.L) for i in range(self.N_sim)]
        self.Sims_created = True 
    
    def N_pool(self, N_list):
        self.Sims = [Simulation(N = N_list[i], limites = self.limite,
                     dispersion = self.dispersion, bruit = self.bruit,
                     vision = self.L) for i in range(self.N_sim)]
        self.Sims_created = True 


    def full_Simulate(self, n_step):
        self.simulated = True
        return [self.Sims[i].Simulate(n_step) for i in range(self.N_sim)]
