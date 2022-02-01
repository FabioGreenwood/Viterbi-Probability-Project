# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:31:32 2022

@author: fabio
"""



#%% Imports

from datetime import datetime
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
import plotly.express as px
import random


#%% functions

class markov_chain:
    def __init__(self):
        self.emissions_matrix = []
        self.transitions_matrix = []
        self.initial_distribution = []
        self.visible_states = None
        self.hidden_states = 6
        self.average_0_transition_links = 0 #count of the average number of links with a zero probability per state
        self.average_0_emission_links = 0 #count of the average number of links with a zero probability per state

    def generate_transition_matrix_step_1(self):
        
        # this functions produces a random transition matrix by assigning each element with a random number then normalising each value to ensure that the
        # stocastic condition is met (each row totals to one). 
        # There is also an if statment to randomly assign variables to zero
        # There is also a check that all states commincate, otherwise the transition matrix is regenerated
        
        requirements_statisfied = False
        requirements_statisfied_1 = False
        row_number = 0
        self.transitions_matrix = []
        while requirements_statisfied == False:
            
            
            
            for row_num in range(0,self.hidden_states):
            
                #popualate a single row
                while requirements_statisfied_1 == False:
                    transitions_matrix_ith_row = np.array([])
                    for col_num in range(0,self.hidden_states):
                        print(col_num)
                        if random.random() < (self.average_0_transition_links/ self.hidden_states):
                            transitions_matrix_ith_row = transitions_matrix_ith_row.append(0)
                        else:
                            transitions_matrix_ith_row = transitions_matrix_ith_row.append(random.random())
                            #fesibile row if exit route out of state exists
                            if not(row_number == col_num):
                                requirements_statisfied_1 = True
                    
                    #Normalisation
                    for i in range(0,self.hidden_states):
                        transitions_matrix_ith_row[col_num] = transitions_matrix_ith_row[col_num] / transitions_matrix_ith_row.sum
                    
                    if row_num == 0:
                        self.transitions_matrix = transitions_matrix_ith_row
                    else:
                        self.transitions_matrix = np.vstack([self.transitions_matrix,transitions_matrix_ith_row])
                    print("Hello")
            
            
MC = markov_chain()
            
MC.generate_transition_matrix_step_1()


            
        
    

    






