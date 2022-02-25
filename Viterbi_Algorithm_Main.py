# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:31:32 2022

@author: fabio
"""

#%% Start-Up Run

runcell('imports', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')
runcell('general parameters', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')
runcell('functions', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')
runcell('script', 'C:/Users/fabio/OneDrive/Documents/Studies/Viterbi_Project/Viterbi-Probability-Project/Viterbi_Algorithm_Main.py')

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


#%% script

"""
#Generate the markov chain matrixes
MC = markov_chain(chain_length = 3, visible_states_qty = 4, hidden_states_qty = 6, average_0_transition_links = 2.5, average_0_emission_links = 1)            
MC.transitions_matrix = generate_transition_matrix_step(MC.hidden_states_qty, MC.average_0_transition_links)
MC.emissions_matrix = generate_emissions_matrix_step(MC.hidden_states_qty, MC.visible_states_qty, MC.average_0_emission_links)
MC.initial_distribution = return_stationary_distribution(MC.transitions_matrix)

test_results = return_the_accuracy_stats_on_entered_MC_without_hidden_or_visible_states(MC, 10)
"""


#%%%


MC_char_dict = {"chain_length" : 3, "visible_states_qty" : 4, "hidden_states_qty" : 6, "average_0_transition_links" : 2.5, "average_0_emission_links" : 1}

overall_results = run_a_study_on_n_generated_MCs(1, 5, MC_char_dict)

    
#%% function under development



def run_a_study_on_n_generated_MCs(MC_qty, reps_per_MC, MC_char_dict):
    """this function will generate X number of MCs based on your input criteria and compare the viteri/DOE_brute_force's accuracies at predicting the hidden state
    For each generated MC it will repeat the study the user specified number of times  
    """
    
    stat_names_to_append = ["most_likely_probability", "viterbi_probability"]
    stat_names_to_find_the_mean = ["viterbi_correct", "viterbi_correct_rate", "viterbi_correct_history", "most_likely_correct", "most_likely_correct_rate", "most_likely_correct_history"]
    
    study_results = repitition()
    
    study_results.viterbi_correct         = 0 
    study_results.viterbi_correct_rate    = 0
    study_results.viterbi_correct_history = np.zeros(MC_char_dict["chain_length"])
       
    study_results.most_likely_correct         = 0
    study_results.most_likely_correct_rate    = 0
    study_results.most_likely_correct_history = np.zeros(MC_char_dict["chain_length"])
    
    study_results.most_likely_probability = np.array([])
    
    for i in range(0, MC_qty):
        MC = markov_chain(MC_char_dict["chain_length"], MC_char_dict["visible_states_qty"], MC_char_dict["hidden_states_qty"], MC_char_dict["average_0_transition_links"], MC_char_dict["average_0_emission_links"])            
        MC.transitions_matrix = generate_transition_matrix_step(MC.hidden_states_qty, MC.average_0_transition_links)
        MC.emissions_matrix = generate_emissions_matrix_step(MC.hidden_states_qty, MC.visible_states_qty, MC.average_0_emission_links)
        MC.initial_distribution = return_stationary_distribution(MC.transitions_matrix)
    
        MC_reps_results = return_the_accuracy_stats_on_entered_MC_without_hidden_or_visible_states(MC, reps_per_MC)
        
        # append the value of everystat set to append
        for stat_name in stat_names_to_append:
            new_value = np.append(getattr(study_results, stat_name), getattr(MC_reps_results, stat_name))
            setattr(study_results, stat_name, new_value )
                
            
            #study_results.set_var(stat_name, np.append(study_results.get_var(stat_name), MC_reps_results.get_var(stat_name)))
                                  
        # add the value of everystat set to find the mean of (normilisation later)
        for stat_name in stat_names_to_append:
            original_value = study_results.get_var(stat_name)
            new_value = original_value + MC_reps_results.get_var(stat_name)
            study_results.set_var(stat_name, new_value)
            #study_results.set_var(stat_name, study_results.get_var(stat_name) += MC_reps_results.get_var(stat_name))
    
    # normalise all stats set to find the mean of
    for stat_name in stat_names_to_append:
        study_results.set_var(stat_name, study_results.get_var(stat_name) / MC_qty)

    return study_results




def return_the_accuracy_stats_on_entered_MC_without_hidden_or_visible_states(markov_chain_input, number_of_reps):
    
    """this function returns various average stats (based on the contents of the repitition class) for repeated runds on a markov chain
    
    in the current make up a MC with a defined transition and emssions matrix is entered and the hidden/visibles states are regenerated and predicted each time
    
    """
    
    MC = markov_chain_input
    
    MC_reps_results  = repitition()
   
    MC_reps_results.viterbi_correct         = 0 
    MC_reps_results.viterbi_correct_rate    = 0
    MC_reps_results.viterbi_correct_history = np.zeros(MC.chain_length)
   
    MC_reps_results.most_likely_correct         = 0
    MC_reps_results.most_likely_correct_rate    = 0
    MC_reps_results.most_likely_correct_history = np.zeros(MC.chain_length)

    for i in range(0, number_of_reps):
        #Generate markov chain states
        rep1 = repitition()
        rep1.markov_chain = MC
        rep1.hidden_states = generate_hidden_markov_chain(rep1.markov_chain)
        rep1.visible_states = generate_visible_markov_chain(rep1.markov_chain, rep1)
        rep1.viterbi_prediction = viterbi_algorithm(MC.transitions_matrix, MC.emissions_matrix, MC.initial_distribution, rep1.visible_states)
        rep1.most_likely_prediction, rep1.most_likely_probability, rep1.P_Y = full_DOE_scan_of_hidden_MC_chain_approach_dev(rep1.visible_states, MC.transitions_matrix, MC.emissions_matrix, MC.initial_distribution)
        
        # rate the predictions
        output_a, output_b, output_c = return_rating_stats_on_prediction(rep1.hidden_states, rep1.viterbi_prediction)
        rep1.viterbi_correct = output_a
        rep1.viterbi_correct_rate = output_b
        rep1.viterbi_correct_history = output_c
        
        output_d, output_e, output_f = return_rating_stats_on_prediction(rep1.hidden_states, rep1.most_likely_prediction)
        rep1.most_likely_correct = output_d
        rep1.most_likely_correct_rate = output_e
        rep1.most_likely_correct_history = output_f
        
        rep1.viterbi_probability = return_P_Y_X_P_X_of_markov_chain(rep1.visible_states, MC.transitions_matrix, MC.emissions_matrix, MC.initial_distribution, rep1.viterbi_prediction) / rep1.P_Y 
        
        
        #log results of repitition
        MC_reps_results.most_likely_probability = np.array( MC_reps_results.most_likely_probability, rep1.most_likely_probability)
        MC_reps_results.viterbi_probability     = np.array( MC_reps_results.viterbi_probability, rep1.viterbi_probability)
       
        MC_reps_results.viterbi_correct         += rep1.viterbi_correct 
        MC_reps_results.viterbi_correct_rate    += rep1.viterbi_correct_rate 
        MC_reps_results.viterbi_correct_history += rep1.viterbi_correct_history
       
        MC_reps_results.most_likely_correct         += rep1.most_likely_correct 
        MC_reps_results.most_likely_correct_rate    += rep1.most_likely_correct_rate 
        MC_reps_results.most_likely_correct_history += rep1.most_likely_correct_history 
   
    #normalise results
    MC_reps_results.viterbi_correct         += rep1.viterbi_correct / number_of_reps
    MC_reps_results.viterbi_correct_rate    += rep1.viterbi_correct_rate / number_of_reps
    MC_reps_results.viterbi_correct_history += rep1.viterbi_correct_history / number_of_reps
   
    MC_reps_results.most_likely_correct         += rep1.most_likely_correct / number_of_reps
    MC_reps_results.most_likely_correct_rate    += rep1.most_likely_correct_rate / number_of_reps 
    MC_reps_results.most_likely_correct_history += rep1.most_likely_correct_history / number_of_reps

    return  MC_reps_results 









#%%

aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
print(np.random.choice(aa_milne_arr, p=[0.5, 0.1, 0.1, 0.3]))

print()


#%% functions

class markov_chain:
    def __init__(self, chain_length = 3, visible_states_qty = 4, hidden_states_qty = 6, average_0_transition_links = 2.5, average_0_emission_links = 1 ):
        self.chain_length = chain_length
        self.emissions_matrix = []
        self.transitions_matrix = []
        self.initial_distribution = []
        self.visible_states_qty = visible_states_qty
        self.hidden_states_qty = hidden_states_qty
        self.average_0_transition_links = average_0_transition_links #count of the average number of links with a zero probability per state
        self.average_0_emission_links = average_0_emission_links #count of the average number of links with a zero probability per state


class repitition:
    def __init__(self):
        self.hidden_states = np.array([])
        self.visible_states = np.array([])
        
        self.viterbi_prediction = []
        self.viterbi_probability = None
        self.viterbi_correct = None
        self.viterbi_correct_rate = None
        self.viterbi_correct_history = None
        
        self.most_likely_prediction = []
        self.most_likely_probability = None
        self.most_likely_correct = None
        self.most_likely_correct_rate = None
        self.most_likely_correct_history = None
        self.P_Y = None
        
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
            previous_dis = test_distribution 
            try:
                test_distribution = np.matmul(test_distribution, input_transitions_matrix)
            except ValueError as err:
                print("FG_BUG Fabio you need to understand why this crash happens half of the time")
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
    



def generate_hidden_markov_chain(input_markov_chain_obj):
    MC_ = input_markov_chain_obj
    output_mc = np.array([np.random.choice(range(0,MC_.hidden_states_qty), p=MC_.initial_distribution)])
    for period in range(1, MC_.chain_length):
        #current_state = np.zeros(MC_.hidden_states_qty)
        #current_state[output_mc[period-1]] = 1
        #next_step_prob = np.matmul(current_state, MC_.transitions_matrix)
        next_step_prob = MC_.transitions_matrix[output_mc[period-1]]
        new_state = np.random.choice(range(0,MC_.hidden_states_qty), p=next_step_prob)
        output_mc = np.append(output_mc, new_state) 
    return output_mc




def generate_visible_markov_chain(input_markov_chain_obj, rep_input):
    output_vis_mc = np.array([])
    MC_ = input_markov_chain_obj
    for period in range(0,len(rep_input.hidden_states)):
        next_step_prob = MC_.emissions_matrix[rep_input.hidden_states[period]]
        new_vis_state = np.random.choice(range(0,input_markov_chain_obj.visible_states_qty), p=next_step_prob)
        output_vis_mc = np.append(output_vis_mc, new_vis_state) 
    return output_vis_mc


def viterbi_algorithm(transitions_matrix, emissions_matrix, initial_distribution, visible_states):
   
    
    
    probabilities = np.empty([np.size(emissions_matrix, 0), len(visible_states)])
    probabilities[:] = np.NaN
    most_likely_hidden_states = np.array([])
    
    # calculate the probability of each hidden state for the first time step
    for hidden in range(0, np.size(emissions_matrix,0)):
        try:
            probabilities[hidden, 0] = initial_distribution[hidden] * emissions_matrix[hidden, int(visible_states[0])]
        except IndexError as err:
            print("FG_DEBUG")
    
    #calculation of the every other time step
    for time_step in range(1, len(visible_states)):
        for current_hidden in range(0, np.size(emissions_matrix,0)):
            considered_possibilities = np.array([])
            for previous_hidden in range(0, np.size(emissions_matrix,0)):
                additional_prob = probabilities[previous_hidden, time_step - 1] * transitions_matrix[previous_hidden, current_hidden] * emissions_matrix[current_hidden, int(visible_states[time_step-1])]
                considered_possibilities = np.append(considered_possibilities, additional_prob)
            probabilities[current_hidden, time_step] = considered_possibilities.max()

    
    #log the model likely hidden states from each time step
    for time_step in range(0, len(visible_states)):
        time_step_probs = probabilities[:,0]
        most_likely_state = np.argmax(time_step_probs)
        most_likely_hidden_states = np.append(most_likely_hidden_states, int(most_likely_state))    
    
    return most_likely_hidden_states


def permutations_generator(time_step_qtp, possible_states_qty):
    # This method returns every permutation of a give markov chain, to be later used by brute force testing of a state


    output_as_array = []
    output_as_string = []
    perm_qty = possible_states_qty ** time_step_qtp
    for perm_num in range(0, perm_qty):
        #pdb.set_trace()  
        #print(str(perm_num))
        permutation_makeup = DecimalToNonDecimal(perm_num, possible_states_qty, "", time_step_qtp)
        output_as_string.append(permutation_makeup)
        perm_output = []
        
        for state_num in range(0, len(permutation_makeup)):
            identity = int(permutation_makeup[state_num])
            perm_output.append(identity)
      
        output_as_array.append(perm_output)
    
    return output_as_array, output_as_string

#permutations_generator(2, ["A","B"])[1][1][1]



def DecimalToNonDecimal(num, new_number_base, st, length):
    
    if num >= new_number_base:
        
        st1 = DecimalToNonDecimal(num // new_number_base, new_number_base, st, length -1)
        st2 = str(num % new_number_base)
        #pdb.set_trace()
        st = st1 + st2
    else:
      st = str(num)

    if len(st) < length:
      for i in range(len(st), length):
        #pdb.set_trace()
        st = '0' + st

    return st
    #print(num % new_number_base, end = '')


def fg_counter(counter_value, total_iterations_qty, number_of_updates_required_for_total_run = 10, start = datetime.now(), update_counter = False, report = ""):
    #""" you will need to deploy a counter variable outside this method """
    if float(counter_value) % int(total_iterations_qty / number_of_updates_required_for_total_run) == 0:
        print("-----  " + report)
        print(datetime.now())
        PC = float(float(counter_value) / total_iterations_qty)
        print(str(counter_value) + " / " + str(total_iterations_qty))
        print(PC)
        print("end estimate @: " + str((datetime.now() - start) * (total_iterations_qty / counter_value) + datetime.now()))
        if update_counter == True:
            return PC


def full_DOE_scan_of_hidden_MC_chain_approach_dev(visible_states, transitions_matrix, emissions_matrix, initial_distribution):
    
    """
    This method works be running through every permutation of the possible hidden states and calculating the independent probability of them happening with the previously stated visible state
    So if Y is the observed markov chain (states) and X is the hidden MC (states). Then this method calculates (for each permutation):
    P(Y|X) * P(X)
    
    The value of P(Y) is calculated by summing every value of P(Y|X) * P(X) for every possible value of P(X) (which are produced by the permutation value)
    The named variable for P(Y) is currently [Total_Probabilities]
    
    
    This description is not complete XXXX
     
    """
                                              
    #secondary parameters and variables
    length_of_period = len(visible_states)
    num_hidden_states =  len(transitions_matrix)
    potentialPaths_qty = num_hidden_states**length_of_period
    Probabilities = np.array([])
    Total_Probabilities = 0
    
    #generate every potential hidden state MC
    potentialPaths_as_array, potentialPaths_as_string = permutations_generator(length_of_period, num_hidden_states)
    
    #do the following for every permutation (i.e. possible hidden state MC)
    probability_of_permutations = []
    counter = 0
    start = datetime.now()
    for perm in range(0,len(potentialPaths_as_array)):
        
        counter += 1
        #fg_counter(counter, len(potentialPaths_as_array), 1, start, False, "Hello")
        
        #this first for loop appends all the probabilities required to statisty the previous stated visible states and the hidden state of the permutation
        prob_temp = return_P_Y_X_P_X_of_markov_chain(visible_states, transitions_matrix, emissions_matrix, initial_distribution, potentialPaths_as_array[perm])
        
        Probabilities = np.append(Probabilities, prob_temp)
        #the probability of each potential hidden state and visible state conbination are then added to the total to calculate the value of P(Y)
        Total_Probabilities = Total_Probabilities + prob_temp
      
    # Once each each values for P(Y|X)P(X) for each value of X are collected. All the model must do is extract the highest value and devide it by sum{ P(Y|X)P(X), for every value of X} to find P(Y)
    index_of_most_likely_perm = int(np.argmax(Probabilities))
    
    
    ##### convertion of P(Y|X)P(X) in to P(X|Y)
    
    
    """commented this out as its likely an unneeded calculationj"""
    #Probabilities_P_X_Y = np.array([])
    #Probabilities_P_X_Y  = np.empty([len(potentialPaths_as_array)])
    #Probabilities_P_X_Y[:] = np.NaN
    #for perm in range(0,len(potentialPaths_as_array)):
        #Probabilities_P_X_Y[perm] = Probabilities / Total_Probabilities
        
    most_likely_hidden_states = np.array(potentialPaths_as_array[index_of_most_likely_perm])
    P_X_Y_most_likely_perm = Probabilities[index_of_most_likely_perm] / Total_Probabilities
    P_Y = Total_Probabilities
    
    return most_likely_hidden_states, P_X_Y_most_likely_perm, P_Y
    
def return_P_Y_X_P_X_of_markov_chain(visible_states, transitions_matrix, emissions_matrix, initial_distribution, supposed_hidden_states):
    """returns the probability of P(Y|X)P(X) for a given supposed hidden MC and visible MC pair"""
    P_Yn_Xn = np.array([])
    P_Xn_Xn_min1 = np.array([])
    for day in range(0,len(visible_states)):
      
        X_hidden_state = int(supposed_hidden_states[day])
        Y_visible_state  = int(visible_states[day])
    
        if day == 0:
          P_Xn_Xn_min1 = np.append(P_Xn_Xn_min1, initial_distribution[X_hidden_state])
        else:
          P_X_hidden_state_min1 = int(supposed_hidden_states[day - 1])
          P_Xn_Xn_min1 = np.append(P_Xn_Xn_min1, transitions_matrix[P_X_hidden_state_min1, X_hidden_state])  #check this reference works
        P_Yn_Xn = np.append(P_Yn_Xn, emissions_matrix[X_hidden_state][Y_visible_state])
    
    #this second loop then multiplies all the previous probabilities
    prob_temp = 1
    for day in range(0, len(P_Xn_Xn_min1)):
        #pdb.set_trace()
        prob_temp = prob_temp * P_Xn_Xn_min1[day] * P_Yn_Xn[day]
    
    P_Y_X_P_X = prob_temp 
    
    return P_Y_X_P_X


def return_rating_stats_on_prediction(actual_hidden_MC, predicted_hidden_MC):
    time_steps = max(len(actual_hidden_MC), len(predicted_hidden_MC))
    output_completely_correct = None
    output_most_likely_correct_rate = None
    output_most_likely_correct_history = np.array([])
    for i in range(0, time_steps):
        if actual_hidden_MC[i] == predicted_hidden_MC[i]:
            output_most_likely_correct_history = np.append(output_most_likely_correct_history, True)
        else:
            output_most_likely_correct_history = np.append(output_most_likely_correct_history, False)
    
    output_most_likely_correct_rate = output_most_likely_correct_history.sum() / len(output_most_likely_correct_history)
    if output_most_likely_correct_rate == 1:
        output_completely_correct = 1
    else:
        output_completely_correct = 0
    
    return output_completely_correct, output_most_likely_correct_rate, output_most_likely_correct_history 


