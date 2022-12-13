#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:38:39 2022

@author: limm
"""

# /!\ besoin d'un dossier 'Videos_Sheep' dans le working directory pour pouvoir enregistrer des animations (sinon modifier le path dans la fonction)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as m_ani
import matplotlib as mplt
import time
from sklearn import preprocessing

rng = np.random.default_rng()

color_list = list(mplt.colors.TABLEAU_COLORS)


#2 utility functions because our differential equations are in polar coordinates while our data model is cartesian

def cartesianND(t) : #t un tableau nD en coordonnées polaires sur la dernière dimension (n1, n2..., n(d-1), 2)
  return np.array([t.T[0]*np.cos(t.T[1]), t.T[0]*np.sin(t.T[1])]).T
 
def cartesian1D(t) :
  return np.array([t[0]*np.cos(t[1]), t[0]*np.sin(t[1])])
 
# the two following functions are used to produce a FuncAnimation for the sheeps

def frame_producer(Pop, fig, ax, box_size):  
    
    ax.clear()
    im_list = []
    #im_list.append(ax.add_patch(mplt.patches.Circle((Pop[1, 0, 0], Pop[1, 0, 1]), 10, fc = '', ec = 'red')))
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

def film_producer_from_file(filename, box_size = 100):  # 100 is usual box size, beware if it aint
  figure, ax = plt.subplots()
  Pops = np.load(filename)
  ani = m_ani.FuncAnimation(figure, frame_producer, frames = Pops, fargs = (figure, ax, box_size), blit = False)
  return ani
 
    
# A class for managing all sorts of functinos used in simulation, such a interaction matrix, sigmoidish speed evolution,
# interaction function or alignement quantification
 
class Model():
  # n = 2
  # A = np.zeros((n,n))
  # g = lambda x : x
  # eps = lambda x : x
  # D = 1
  # speed = lambda x : 1 + 0*x
 
  def __init__(self, n0, bruit = 0.01, tau = 20, init_A = 'std', speed = 'cste'):
    self.n = n0
    self.A = np.zeros((self.n, self.n))
    if init_A == 'std':
      for i in range(self.n-1):
        self.A[i][i+1] = 1
    elif init_A == 'full':
      self.A += 1/(self.n-1)
      self.A *= (1-np.identity(self.n))
      # self.A[:,0] *= 0
    self.renormer = np.ones_like(self.A.sum(axis = 0))  #les interactions de chaque mouton sont normées à 1...
    #self.renormer[0] = 0                                #...sauf le 1er qui n'interagit pas
     
    # g (mouton1, mouton2)
    self.g = lambda m1, m2 : np.sin( np.angle((m1[0] - m2[0])[0] + (m1[0] - m2[0])[1]*1j) - m1[1,1])
    
    self.D = bruit
    self.eps = lambda size = None : rng.random(size = size) - 0.5
    
    tau_0 = 10  #montée en vitesse (phase de convergence de A) : ~ 100 pour tau_0 = 10 (starts at 0.5)
    tau_1 = 400 #descente en vitesse (phase d'arrêt) : ~ 300->500 pour tau_1 = 400
    if speed == 'sigmoid':
        self.speed = lambda t : (1/(1 + np.exp((tau_0-t)/tau))
                              -  1/(1 + np.exp((tau_1-t)/tau)))
    else :
        self.speed = lambda x : 1 + 0*x



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
        return ((np.cos(calc) - np.identity(self.n)).sum(axis = 1))/(self.n-1)
      elif func == 'clusters':
        grav = np.mean(M[:,0], axis = 0)
        return np.abs((M[:,0] - grav)**2).sum()/self.n
    
    
  def A_evolver(self, M, r = 3, thet_vis = np.pi/4, delta = 0.1, kappa = 10, r_m = 1):  # Permet de faire évoluer la matrice d'interactions
    #r = distance d'équilibre(plus loin = trop loin, plus près = trop près)
    #thet_vis = angle de perception
    #delta = force de l'ajustement
    #kappa = rappel élastique dans le champ de perception
    C = M[:,0, 0] + M[:,0,1]*1j
    xx, yy = np.meshgrid(C, C)
    R = np.absolute(xx - yy)
    T = np.angle(xx-yy)
    #L =((T  < (thet_vis)) & (T > (-thet_vis)))  # être dans l'angle de perception
    
    #etablissement de la liste des moutons contribuants
    indexes = [[]] * (self.n)
    for i in range(self.n):
        actual_useful_index = []
        actual_useful_bloc_angle = []
        R_index = np.argsort(R[i])
        for k in R_index:
            if i == k:
                continue
            taking_k = ((T[i,k]  <= (thet_vis)) & (T[i,k] >= (-thet_vis)))
            j = 0
            while taking_k and  j < (len(actual_useful_index)):
                µ = actual_useful_index[j]
                taking_k *= ((R[i,k]) <= (R[i,µ]) )| ((T[i,k] - T[i,µ]) > actual_useful_bloc_angle[j])
                j += 1
    
            if taking_k:
                actual_useful_index.append(k)
                actual_useful_bloc_angle.append(np.arctan(r_m/R[i,k]))
            
        indexes[i] = actual_useful_index
        
    L = np.zeros_like(self.A)
    for i in range(self.n):
        if indexes[i] != []:
            L[i, indexes[i]] += 1
    self.A += L * delta * kappa * ((R-r) * ((R-r) > 0))
    self.A = preprocessing.normalize(self.A, axis = 0, norm = 'l1') * self.renormer
    
    
    
#A class to manage sheep populations, their evolution step by step, and produce renderings (static or dynamic) for those populations.
#Interesting functions :
        #- Simulate is basically the main() of this whole program
        #- dev_sim is a misc function to run simulations with predetermined parameters and test things; DO NOT USE
        #- evolve is where the physics plays
        #- affiche_population is kind of a legacy function; still useful to look at for debugging limits or initial setups
    
class Simulation():
  # N = 2
  # Population = np.empty((N, 2, 2))   #Troupeau de moutons de forme [mouton1, mouton2...] avec mouton = [position, vitesse] avec position cartésienne et 
  # #vitesse pseudo-polaire (2D)  (la vitesse est tq v = r * exp(i theta))
  # modele = Model(N)
  # verbose = False
  # affichage = False
  # limite = None
  # box_size = 10    # Univers limité par un carré de côté box_size avec un sommet en (0,0)
  def __init__(self, N = 2, verbose = False, affichage = False, limites = 'Align',
               dispersion = 25, box_size = 100, bruit = 0.01, vision = 100,
               init_A = 'std', full_init = None, speed = 'cste'):   #Pour le moment bruit max à 1 et min à 0
    if full_init == None:
        self.verbose = verbose
        self.affichage = affichage
        self.limite = limites
        self.N = N
        self.L= vision #distance caractéristique de vision du mouton
        self.modele = Model(self.N, bruit = bruit, init_A = init_A, speed = speed)
        self.Population = rng.random(size = (N, 2, 2))
        self.box_size = box_size
        self.Population[:,0] = (self.Population[:,0] - 0.5)*dispersion + self.box_size/2
        self.Population[:,1,0] = 1
        self.Population[:,1,1] *= 2*np.pi
    
          
        if self.verbose :
          print(f'Initialisation de la simulation avec {N} moutons et des limites de type {limites}')
    else:
        self.verbose = verbose
        self.affichage = affichage
        self.limite = limites
        self.N = N
        self.L= vision #distance caractéristique de vision du mouton
        self.modele = Model(self.N, bruit = 0, init_A = 'std')
        self.Population = full_init
        self.box_size = box_size
  def evolve(self, evolve_A = False, step = 0):
    #stock_pop = self.Population.copy()
    
    if evolve_A:
        self.modele.A_evolver(self.Population)
        self.A_stocker[step] += self.modele.A
    
    self.Population[:,1,1] += (self.modele.A * self.modele.G(self.Population, L = self.L)).sum(axis = 0
                   ) + 2*np.pi*self.modele.D*self.modele.eps(size = (self.N))  #bruit max à 1, amplifié quand les autres sont loin ? mvt limité à pi/2?

    self.Population[:,1,1] %= 2*np.pi

    self.Population[:,0] += self.modele.speed(step) * cartesianND(self.Population[:,1])
    
    if self.limite == 'BVK':
      self.Population[:,0] %= self.box_size  #limite : boîte de taille 100x100
    
    if self.limite == 'Align':      #WIP, almost working  #toujours des problèmes visibles
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
  def Simulate(self, n_step, animate = False, anim_name = 'Sheep_Sim_Test',
               evolve_A = False, return_traj = False):
    filename = 'Videos_Sheep/' + anim_name + '.mp4'
    if evolve_A:
      self.A_stocker = np.zeros((n_step, self.N, self.N))
    
    if self.verbose:
      time_0 = time.time()
    
    alignement = np.empty((n_step+1, self.N), dtype = float)
    
    alignement[0] = self.modele.quantify(self.Population)
    
    if animate :
        figure, ax = plt.subplots(figsize = (10,10))
        Pops = np.empty((n_step+1, self.N, 2, 2))
        Pops[0] = self.Population.copy()
    elif return_traj:
        Pops = np.empty((n_step+1, self.N, 2, 2))
        Pops[0] = self.Population.copy()

    for k in range(n_step):
      if (animate) or return_traj:
        Pops[k+1] = self.Population.copy()
      if self.affichage:
        self.affiche_population()
      self.evolve(evolve_A, k)
      alignement[k+1] = self.modele.quantify(self.Population)
      if self.verbose :
          print(f'Evolved step {k}')
      
    print(f'Evolved for {n_step} steps')
    if self.verbose:
      time_1 = time.time()
    
    if animate:
      ani = m_ani.FuncAnimation(figure, frame_producer, frames = Pops, fargs = (figure, ax, self.box_size), blit = False)
      ani.save(filename)
    if self.affichage:
      self.affiche_population()
      
    if self.verbose:
      time_2 = time.time()
      
      print(f'Simulation time : {round(time_1 - time_0, 3)}s; animation time : {round(time_2 - time_1, 3)}s')
      
    if return_traj:
      return alignement, Pops
      
    return alignement

  def dev_sim(self, n_step, animate = True, anim_name = 'Sheep_Sim_Test', evolve_A = True,
              save_traj = False):  #dev func, do not use if unaware about what it does (also very suboptimized)
        if save_traj:
          align, pops = self.Simulate(n_step, animate, anim_name, evolve_A, save_traj)
        else:
          align = self.Simulate(n_step, animate, anim_name, evolve_A, save_traj)
        plt.show()
        for k in range(1,self.N):
            plt.plot(align[:,k], color = color_list[k])
        plt.title('Every sheep alignement')
        plt.show()
        plt.plot(align.mean(axis = 1))
        plt.title('All sheeps alignement')
        plt.show()
        plt.figure(figsize = (15, 10))
        for i in range(self.N):
          plt.subplot(int(self.N/3)+1, 3, i+1)    
          plt.plot([self.A_stocker[k][:,i] for k in range(n_step)])
        plt.tight_layout()
        plt.show()
        
        if save_traj:
          np.save(anim_name + '_traj', pops)
        return align


#A class to manage huge numbers of simulations at once, to make statistical approach easier
      #still WIP  
      
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


    def full_Simulate(self, n_step, get_trajs = True):
        self.simulated = True
        return [self.Sims[i].Simulate(n_step, return_traj = get_trajs) for i in range(self.N_sim)]