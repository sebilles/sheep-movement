#trying some shit out there

import numpy as np
from matplotlib import pyplot as plt

class Mouton():
  positiion = np.array(2, dtype = float) #coords cartésiennes
  vitesse = np.array(2, dtype = float) #coords polaires
  
  def update():
    vit_cart = np.array([vitesse[0] * np.cos(vitesse[1]), vitesse[0] * np.sin(vitesse[1])])
    position += vit_cart
