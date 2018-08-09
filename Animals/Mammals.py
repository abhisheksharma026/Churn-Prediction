# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:43:37 2018

@author: Abhishek 
"""
class Mammals:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']
 
 
    def printMembers(self):
        print('Printing members of the Mammals class')
        for member in self.members:
            print('\t%s ' % member)
