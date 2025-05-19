import numpy as np
from numba import njit
import sys
import time
import pandas as pd
import MCMC_improved_clean as MC
import warnings
import copy
import compressedInputsRead as readInputs
warnings.filterwarnings("ignore")

def realLifeSimulationTime(transition_matrix, rewardVector, utilityVector, *triple):
    if not MC.check_same_size(transition_matrix, rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
    (discountRate, num_steps, num_chains) = triple
    initial_state = 0  # Ensure initial state is appropriate
    stop_state = 28 # This is a terminal state
    base = 1 + discountRate
   
    # BASE MATRICES
    discountMatrix = MC.compute_powers(base, (0,num_steps-1))
    discountMatrix = 1/discountMatrix
    cumulative_probs_cache = MC.precompute_cumulative_probabilities(transition_matrix)

    # RANDOM_NUMBERS
    states = MC.simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state)
    oldStates = copy.deepcopy(states)

    # REWARDS
    rewards = MC.populate_matrix_with_indices(rewardVector, states)
    rewards = rewards @ discountMatrix

    # UTILITIES
    integerUtility = 100*utilityVector  #  as dtype of states is int64
    utilities =  MC.populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
    return np.sum(rewards), np.sum(utilities)

def generate_utility_vector(size):
    # Generate a random matrix of the given size
    matrix = np.random.rand(size, 1)
    
    # Normalize each row to make it a probability distribution
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_matrix = matrix / row_sums  # Broadcasting division
    
    return transition_matrix


# To unpack distinct distributions triple = (discount_rate, num_steps, num_chains)
# Here the {key: values} could be anything that produces different transitions
# This is general, can be used to generate all the different cases.
def parallelSimulation(*triple, parameters):
    result = {}
    for key,value in parameters.items():
        transition_matrix, rewardVector, utilityVector = value

        result[key] = realLifeSimulation(transition_matrix, rewardVector, utilityVector,*triple)
    return result




if __name__ ==  "__main__":
    file = r"C:\Users\sovan\Box\Sovann Linden's Files\Cost-effectiveness\Inputs\Inputs_CEA_v3.xlsx"
    df1 = pd.read_excel(file, sheet_name='ActuarialTables')  
    readInputs.generate_prob_death(df1)
    df2 = pd.read_excel(file, sheet_name='AgeVector')  
    readInputs. generate_age_vector(df2,1000)
    
    




    probDeath = np.array([0,0.01, 0.02])
    transition_matrix = np.array([[0.028, 0.65, 0.322 ],[0.1, 0.5, 0.4],[0.1, 0.5, 0.4]])
    transition_matrix = MC.generate_transition_matrixAgeDistribution(transition_matrix, 3, probDeath,1)
    print(transition_matrix)
  
    
   


   