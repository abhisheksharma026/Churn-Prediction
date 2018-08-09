# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:51:36 2018

@author: Abhishek 
"""

# Import classes from your brand new package
from Animals.Mammals import Mammals
from Animals.Birds import Birds
 
# Create an object of Mammals class & call a method of it
myMammal = Mammals()
myMammal.printMembers()
 
# Create an object of Birds class & call a method of it
myBird = Birds()
myBird.printMembers()