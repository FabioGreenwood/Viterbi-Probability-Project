# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:31:32 2022

@author: fabio
"""

#%% Start-Up Run

runcell('imports', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')
runcell('general parameters', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')
runcell('functions', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')
runcell('Script', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')

#%% imports

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
from numpy import linalg as LA
from scipy.linalg import eig 


#%% general parameters

chain_length = 6


#%% Script
            
MC = markov_chain()            
MC.transitions_matrix = generate_transition_matrix_step(MC.hidden_states_qty, MC.average_0_transition_links)
MC.emissions_matrix = generate_emissions_matrix_step(MC.hidden_states_qty, MC.visible_states_qty, MC.average_0_emission_links)
MC.initial_distribution = return_stationary_distribution(MC.transitions_matrix)

rep1 = repitition()
rep1.markov_chain = MC
rep1.hidden_states = generate_hidden_markov_chain(MC, chain_length)
rep1.visible_states = generate_visible_markov_chain(MC, rep1)




#%%










#%%

aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
print(np.random.choice(aa_milne_arr, p=[0.5, 0.1, 0.1, 0.3]))

print()


#%% functions

class markov_chain:
    def __init__(self):
        self.emissions_matrix = []
        self.transitions_matrix = []
        self.initial_distribution = []
        self.visible_states_qty = 4
        self.hidden_states_qty = 6
        self.average_0_transition_links = 2.5 #count of the average number of links with a zero probability per state
        self.average_0_emission_links = 1 #count of the average number of links with a zero probability per state


class repitition:
    def __init__(self):
        self.hidden_states = np.array([])
        self.visible_states = np.array([])
        
        self.viterbi_prediction = []
        self.viterbi_probability = None
        self.viterbi_correct = None
        
        self.most_likely_prediction = []
        self.most_likely_probability = None
        self.most_likely_correct = None
        
        self.markov_chain = None



    
def generate_emissions_matrix_step(input_hidden_states_qty, input_visible_states_qty, input_average_0_emission_links):
    
    # this functions produces a random emissions matrix by assigning each element with a random number then normalising each value to ensure that the
    # stocastic condition is met (each row totals to one). 
    # There is also an if statment to randomly assign variables to zero
    # There is also a check that all visible states has at least one hidden state attached to them commincate, otherwise the matrix is regenerated
    
    requirements_statisfied = False
    requirements_statisfied_1 = False
    row_number = 0
    
    while requirements_statisfied == False:
        
        emissions_matrix = np.array([])
        for row_num in range(0,input_hidden_states_qty):
            requirements_statisfied_1 = False
            #popualate a single row
            while requirements_statisfied_1 == False:
                emissions_matrix_ith_row = np.array([])
                emissions_for_hidden_state_in_question = 0
                for col_num in range(0,input_visible_states_qty):
                    #print(col_num)
                    if random.random() < (input_average_0_emission_links / input_visible_states_qty):
                        emissions_matrix_ith_row = np.append(emissions_matrix_ith_row, 0)
                    else:
                        emissions_for_hidden_state_in_question += 1
                        emissions_matrix_ith_row = np.append(emissions_matrix_ith_row, random.random())
                    #Check to ensure that each hidden state, links to at least 2 visible states
                    if emissions_for_hidden_state_in_question > 1:
                        requirements_statisfied_1 = True
                            
                
                #Normalisation
                row_sum = emissions_matrix_ith_row.sum()
                for i in range(0,input_visible_states_qty):
                    emissions_matrix_ith_row[i] = emissions_matrix_ith_row[i] / row_sum
                
                if row_num == 0:
                    emissions_matrix = emissions_matrix_ith_row
                else:
                    emissions_matrix = np.vstack((emissions_matrix,emissions_matrix_ith_row))
                #print("Hello")
        requirements_statisfied = check_that_each_visible_state_has_at_least_two_hidden_states(emissions_matrix)
    return emissions_matrix




def generate_transition_matrix_step(input_hidden_states_qty, input_average_0_transition_links):
    
    # this functions produces a random transition matrix by assigning each element with a random number then normalising each value to ensure that the
    # stocastic condition is met (each row totals to one). 
    # There is also an if statment to randomly assign variables to zero
    # There is also a check that all states commincate, otherwise the transition matrix is regenerated
    
    requirements_statisfied = False
    requirements_statisfied_1 = False
    row_number = 0
    
    while requirements_statisfied == False:
        
        
        transitions_matrix = np.array([])
        for row_num in range(0,input_hidden_states_qty):
            requirements_statisfied_1 = False
            #popualate a single row
            while requirements_statisfied_1 == False:
                transitions_matrix_ith_row = np.array([])
                for col_num in range(0,input_hidden_states_qty):
                    #print(col_num)
                    if random.random() < (input_average_0_transition_links/ input_hidden_states_qty):
                        transitions_matrix_ith_row = np.append(transitions_matrix_ith_row, 0)
                    else:
                        transitions_matrix_ith_row = np.append(transitions_matrix_ith_row, random.random())
                        #fesibile row if exit route out of state exists
                        if not(row_number == col_num):
                            requirements_statisfied_1 = True
                
                #Normalisation
                row_sum = transitions_matrix_ith_row.sum()
                for i in range(0,input_hidden_states_qty):
                    transitions_matrix_ith_row[i] = transitions_matrix_ith_row[i] / row_sum
                
                if row_num == 0:
                    transitions_matrix = transitions_matrix_ith_row
                else:
                    transitions_matrix = np.vstack((transitions_matrix,transitions_matrix_ith_row))
                #print("Hello")
        requirements_statisfied = check_all_states_in_same_communication_class_transmission(transitions_matrix)
    return transitions_matrix




def return_stationary_distribution(input_transmission_matrix):
    S, U = eig(input_transmission_matrix.T)
    stationary_temp = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary_temp = stationary_temp / np.sum(stationary_temp)
    stationary = np.array([])
    for j in range(0,len(stationary_temp)):
        stationary = np.append(stationary, float(stationary_temp[j]))
    return stationary




def check_all_states_in_same_communication_class_transmission(input_transitions_matrix):
    #Works by ensuring there is no zero value in the resulting vector -> [pi * trannsition matrix ^ x] where pi is an initial distri
    length_of_matrix = len(input_transitions_matrix)
    all_states_in_commincation = True
    for row in range(0,length_of_matrix):
        test_distribution = np.zeros(6)
        test_distribution[row] = 1
        for i in range(0,length_of_matrix + 2):
            test_distribution = np.matmul(test_distribution, input_transitions_matrix)
        if  (test_distribution == 0).sum() > 0:
            all_states_in_commincation = False
    
    
    return all_states_in_commincation



    
def check_that_each_visible_state_has_at_least_two_hidden_states(input_emissions_matrix):
    pass_crit = True
    for col in range(0, len(input_emissions_matrix[0])):
        non_zero_count = 0
        for row in range(0, len(input_emissions_matrix)):
            if input_emissions_matrix[row][col] > 0:
                non_zero_count += 1
        if non_zero_count < 2:
            pass_crit = False
    return pass_crit  
        
        
    print("Hello")
    print("Hello")
    



def generate_hidden_markov_chain(input_markov_chain_obj, chain_length):
    MC_ = input_markov_chain_obj
    output_mc = np.array([np.random.choice(range(0,MC_.hidden_states_qty), p=MC_.initial_distribution)])
    for period in range(1,chain_length):
        #current_state = np.zeros(MC_.hidden_states_qty)
        #current_state[output_mc[period-1]] = 1
        #next_step_prob = np.matmul(current_state, MC_.transitions_matrix)
        next_step_prob = MC_.transitions_matrix[output_mc[period-1]]
        new_state = np.random.choice(range(0,MC.hidden_states_qty), p=next_step_prob)
        output_mc = np.append(output_mc, new_state) 
    return output_mc




def generate_visible_markov_chain(input_markov_chain_obj, rep_input):
    output_vis_mc = np.array([])
    MC_ = input_markov_chain_obj
    for period in range(0,len(rep_input.hidden_states)):
        next_step_prob = MC_.emissions_matrix[rep_input.hidden_states[period]]
        new_vis_state = np.random.choice(range(0,MC.visible_states_qty), p=next_step_prob)
        output_vis_mc = np.append(output_vis_mc, new_vis_state) 
    return output_vis_mc

#%% Viterbi from previous


# TransitionProbabilities
p_ss = 0.8
p_sr = 0.2
p_rs = 0.4
p_rr = 0.6

# Initial Probabilities
p_s = 2/3
p_r = 1/3

# Emission Probabilities
p_sh = 0.8
p_sg = 0.2
p_rh = 0.4
p_rg = 0.6


moods = ['H', 'H', 'G', 'G', 'G', 'H']
probabilities = []
weather = []

if moods[0] == 'H':
  probabilities.append((p_s*p_sh, p_r*p_rh))
else:
  probabilities.append((p_s*p_sg, p_r*p_rg))

for i in range(1, len(moods)):
  yesterday_sunny, yesterday_rainy = probabilities[-1]
  if moods[i] =='H':
    today_sunny = max(yesterday_sunny*p_ss*p_sh, yesterday_rainy*p_rs*p_sh)
    today_rainy = max(yesterday_sunny*p_sr*p_rh, yesterday_rainy*p_rr*p_rh)
    probabilities.append((today_sunny, today_rainy))
  else:
    today_sunny = max(yesterday_sunny*p_ss*p_sg, yesterday_rainy*p_rs*p_sg)
    today_rainy = max(yesterday_sunny*p_sr*p_rg, yesterday_rainy*p_rr*p_rg)
    probabilities.append((today_sunny, today_rainy))

for p in probabilities:
  #pdb.set_trace()
  if p[0] > p[1]:
    weather.append('S')
  else:
    weather.append('R')






#%%
        







