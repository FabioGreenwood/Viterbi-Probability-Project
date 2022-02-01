# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:31:32 2022

@author: fabio
"""



#%% Functions

def permutations_generator(vector_length, possible_states):
# This method returns every permutation of a give vector, given that each element can take any of the given states


  #input variables
  ##Hidden_States = ['S', 'R']
  ##moods = ['H', 'H', 'G', 'G', 'H']

  output1 = []
  output2 = []
  perm_qty = len(possible_states) ** vector_length
  for perm_num in range(0, perm_qty):
    #pdb.set_trace()  
    #print(str(perm_num))
    permutation_makeup = DecimalToNonDecimal(perm_num, len(possible_states), "", vector_length)
    output2.append(permutation_makeup)
    perm_output = []
    
    for state_num in range(0, len(permutation_makeup)):
      identity = int(permutation_makeup[state_num])
      perm_output.append(possible_states[identity])

    output1.append(perm_output)

  return output1, output2

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


#%% Viterbi Algorithm Approach

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

weather  

#%% Full DOE Scan Of Hidden Makrov Chain Approach

# Evaluation of every possible path

#input variables
Hidden_States = ['S', 'R'] # Sunny, Rainy
Visible_States = ['H', 'G'] # Happy, Grumpy
#moods = ['H', 'H', 'G', 'G', 'G', 'H']

moods = ['H', 'H', 'G', 'G', 'G', 'H', 'H', 'G', 'G', 'G', 'H', 'H', 'G', 'G', 'G', 'H', 'H', 'G', 'G', 'G', 'H']


#secondary parameters
Length_of_Period = len(moods)
num_hidden_states =  len(Hidden_States)
potentialPaths_qty = num_hidden_states**Length_of_Period
potentialPaths, potentialPaths_Index = permutations_generator(len(moods), Hidden_States)
Probabilities = []
Total_Probabilities = 0

#calculation variables
probability_of_permutations = []
P_Yn_Xn = []
P_Xn_Xn = []
#P_X1 = []


# TransitionProbabilities
p_ss = 0.8
p_sr = 0.2
p_rs = 0.4
p_rr = 0.6
Trans_Probs = [[p_ss, p_sr],[p_rs, p_rr]]

# Initial Probabilities
p_s = 2/3
p_r = 1/3
Initial_Probs = [p_s, p_r]

# Emission Probabilities
p_sh = 0.8
p_sg = 0.2
p_rh = 0.4
p_rg = 0.6
Emissions_Probs = [[p_sh,p_sg], [p_rh, p_rg]]


for perm in range(0,len(potentialPaths)):
  P_Yn_Xn = []
  P_Xn_Xn = []

  for day_1 in range(0,len(moods)):
    #if str(potentialPaths[perm]) == "['S', 'S', 'R', 'R', 'S']" or str(potentialPaths[perm]) == "['S', 'S', 'S', 'S', 'S']":
    #pdb.set_trace()
    
    X_Index = int(potentialPaths_Index[perm][day_1])
    Y_Index = int(Visible_States.index(moods[day_1],0,len(moods)))
    #print(X_Index)c
    #print(Y_Index)
    #print(step)


    if day_1 == 0:
      P_Xn_Xn.append(Initial_Probs[X_Index])
    else:
      P_X_Index_min1 = int(potentialPaths_Index[perm][day_1 - 1])
      P_Xn_Xn.append(Trans_Probs[P_X_Index_min1][X_Index])
    P_Yn_Xn.append(Emissions_Probs[X_Index][Y_Index])
  prob_temp = 1
  
  for day in range(0, len(P_Xn_Xn)):
    #pdb.set_trace()
    prob_temp = prob_temp * P_Xn_Xn[day] * P_Yn_Xn[day]
  Probabilities.append(prob_temp)
  Total_Probabilities = Total_Probabilities + prob_temp

sorted_probs_asec = Probabilities.copy()
sorted_probs_asec.sort()
#print(Probabilities)
#print(sorted_probs_asec)
optimiumIndex = Probabilities.index(sorted_probs_asec[-1],0,len(sorted_probs_asec))

#####separate calculation of P(Y)
Total_Probabilities2 = 1
for day_2 in range(0,len(moods)):
  # P(Y_n) = P(Y_n|X_n)P(X_n) + P(Y_n|X_n^c)P(X_n^c)
  # P(Y) = Joint P(Y_n)
  Y_Index = int(Visible_States.index(moods[day_2],0,len(moods)))
  #Total_Probabilities2 = Total_Probabilities2 * 


prob = Probabilities[optimiumIndex] / Total_Probabilities

print("most likely permutation is: " + str(potentialPaths[optimiumIndex]))
prob_print = str("%.2f" % prob)
print("with a prob of: ", prob_print)
print("this is out of a total " + str(len(potentialPaths)) + str(" potential permutations"))

#print(sorted_probs_asec[1])
#print(Probabilities[optimiumIndex])












