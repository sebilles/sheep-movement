#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:42:27 2022

@author: chanzc
"""

from classes_def import *

#Crée trois simulations avec différents paramètres et les simule en enregistrant un film
#Paramètres de Simulation:
    #n : nombre de moutons (première variable)
    
    # bruit : le bruit sur l'équation différentielle (entre 0 et 1)
    # vision: la distance de vision des moutons (distance caractéristique du flou gaussien)
    # box_sizee : taille de l'enclos
    # init_A : 'std' ou 'full', initialise la matrice d'interactions sur une diagonale ou isotrope
    # dispersion : distance caractéristique entre les moutons à l'initialisation
    # limites : 'Align' ou 'BVK' - conditions aux bors de l'enclos d'alignement vs périodiques
    # full_init : initialisation déterminée des moutons (passer la matrice (N, 2, 2) des positions + vitesses initiales)
    
    #verbose : affiche quelques textes supplémentaires, notamment de durée de run
    #affichage : affiche chaque étape de la run

#Paramètres de Simulate:
    #n_step : nombre d'étapes à simuler (première variable)
    
    #animate : booléen, si True : crée une animation et l'enregistre
    #anim_name : nom de l'animation pour la sauvegarde
    #evolve_A : booléen, si True : fait évoluer la matrice d'interaction des moutons au cours de la simulation
    #return_traj : booléen, si True : renvoie toutes les positions et vitesses  à la fin de la simulation
        # Simulate renvoie par défaut l'alignement Qi de chaque mouton à chaque étape; en cas de return_traj, 
        # préciser "align, traj = Simulate(...)"

z = Simulation(4, bruit = 0)
a = z.Simulate(50, animate = True, anim_name = 'Simulation_Moutons_bruit_nul', evolve_A = False)


zp = Simulation(4, bruit = 0.1)
ap = zp.Simulate(50, animate = True, anim_name = 'Simulation_Moutons_bruit_faible', evolve_A = False)


zt = Simulation(4, bruit = 0.7)
at = zt.Simulate(50, animate = True, anim_name = 'Simulation_Moutons_bruit_fort', evolve_A = False)



#Affiche le Q moyen pour les trois simulations
plt.figure()

plt.plot(a.mean(axis =1), label = 'D = 0')
plt.plot(ap.mean(axis =1), label = 'D = 0.1')
plt.plot(at.mean(axis =1), label = 'D = 0.7')

plt.xlabel('Temps (étapes)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Q moyen', fontsize = 12, fontweight = 'bold')
plt.title('Alignement de 4 moutons en fonction du bruit', fontsize = 12, fontweight = 'bold')
plt.ylim(top = 1.05)
plt.legend()
plt.show()