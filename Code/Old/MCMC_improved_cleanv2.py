import numpy as np
from numba import njit
import sys
import time
import pandas as pd
import compressedInputsRead as readInputs

  # For sizing num_states: 20, num_chains: high, num_steps: 50
  # initial state here is common for all people
  # stop state is also common here for all people
import warnings
import copy
warnings.filterwarnings("ignore")


#########   PRECOMPUTE PROBABILITIES ##########

@njit
def precompute_cumulative_probabilities(transition_matrix):
    num_states = transition_matrix.shape[0]
    cumulative_probs_cache = np.zeros((num_states, num_states))
    for s in range(num_states):
        cumulative_probs_cache[s, 0] = transition_matrix[s, 0]
        for j in range(1, num_states):
            cumulative_probs_cache[s, j] = cumulative_probs_cache[s, j - 1] + transition_matrix[s, j]
    return cumulative_probs_cache
# This was numerically tested!!
@njit
def precompute_cumulative_probabilitieswithAge(transition_matrix):
    num_states = transition_matrix.shape[0]
    num_ages = int(transition_matrix.shape[1]/num_states)
   
    for x in range(0, num_ages):
        a = precompute_cumulative_probabilities(transition_matrix[:,x*num_states:(x+1)*num_states])
        if x == 0:
            cumulative_probs_cache = a
        else:
            cumulative_probs_cache = np.append(cumulative_probs_cache,a ,axis = 1)
    

    return cumulative_probs_cache


#########   RANDOM NUMBER GENERATION ##########

@njit
def simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state):
    num_states = transition_matrix.shape[0]
    states = np.zeros((num_chains, num_steps), dtype=np.int64)
    states[:, 0] = initial_state

    # This generates all the random numbers we need
    random_numbers = np.random.rand(num_chains, num_steps - 1)

    # Here we actually convert the random draw into a state sequence
    for t in range(1, num_steps):
        for i in range(num_chains):
            if states[i, t-1] == stop_state:
                states[i, t] = stop_state
                continue

            current_state = states[i, t-1]
            cumulative_probs = cumulative_probs_cache[current_state] 

            # This is a neat trick to convert a random draw with a list of probabilities into a linear search problem
            states[i, t] = np.searchsorted(cumulative_probs, random_numbers[i, t-1])

    return states
@njit
def simulate_with_cache_ageDistribution(cumulative_probs_cache, ageVector, *quad):
    initial_state, num_steps, num_chains, stop_state = quad
    num_states = cumulative_probs_cache.shape[0]
    states = np.zeros((num_chains, num_steps), dtype=np.int64)
    states[:, 0] = initial_state

    # This generates all the random numbers we need
    random_numbers = np.random.rand(num_chains, num_steps - 1)

    # Here we actually convert the random draw into a state sequence
    print("t,y,age,state,next_state, translation, random")
    for t in range(1, num_steps):
        for i in range(num_chains):
            if states[i, t-1] == stop_state:
                states[i, t] = stop_state
                continue

            current_state = states[i, t-1]
            current_age = ageVector[i] + t +18 -1 
            # For age 50, we want 50-18th +1 = 32nd block in expanded transition matrix
            translationIndex= num_states*(current_age-18) 
            cumulative_probs = cumulative_probs_cache[current_state, translationIndex: translationIndex+num_states] 
            arr = cumulative_probs
            if len(arr) == 0 or arr[-1] != 1:  # Early check for empty array or last element
                print(False)
                print(i,t, "Error 1", arr)
            for j in range(1, len(arr)):
                if arr[j] < arr[j-1] or arr[j] <= 0:  # Check for increasing and positive
                    print(i,t, "Error 2", arr)




            # This is a neat trick) to convert a random draw with a list of probabilities into a linear search problem
            states[i, t] = np.searchsorted(cumulative_probs, random_numbers[i, t-1])
            
           # print(t,i,current_age, current_state, states[i,t], translationIndex, np.round(random_numbers[i, t-1],3),[np.round(x,3) for x in cumulative_probs])
                

    return states



#########   UTILITY FUNCTIONS AND TESTS ##########



def generate_transition_matrix(size):
    # Generate a random matrix of the given size
    matrix = np.random.rand(size, size)
    
    # Normalize each row to make it a probability distribution
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_matrix = matrix / row_sums  # Broadcasting division
    
    return transition_matrix
def generate_utility_vector(size):
    # Generate a random matrix of the given size
    matrix = np.random.rand(size, 1)
    
    # Normalize each row to make it a probability distribution
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_matrix = matrix / row_sums  # Broadcasting division
    
    return transition_matrix
def generate_reward_vector(size):
    # Generate a random matrix of the given size
    matrix = np.random.rand(size, 1)
    
    return matrix
def generate_age_vector(size):
    return np.random.randint(18, 101, size)
@njit 
def compute_powers(base, exponent_range):
    # Create an array of exponents
    exponents = np.arange(exponent_range[0], exponent_range[1] + 1)
    
    # Compute the powers using broadcasting
    powers = base ** exponents
    
    return powers
def check_same_size(*arrays):
    # Get the shape or size of the first array
    first_shape = arrays[0].shape if len(arrays[0].shape) > 1 else (len(arrays[0]),)
    
    # Check if all other arrays (matrices or vectors) have the same size
    for array in arrays[1:]:
        current_shape = array.shape if len(array.shape) > 1 else (len(array),)
        if current_shape != first_shape:
            print(current_shape, first_shape)
            return False  # If any array is not the same size, return False
    return True 
def check_if_array(matrices):
    # Check if each element in the list is a NumPy array
    return all(isinstance(matrix, np.ndarray) for matrix in matrices)
@njit
def populate_matrix_with_indices(v, matrix):
    shape = matrix.shape
    oldMatrix = matrix #as we change value of matrix
    
    for i in range(shape[0]):  # Iterate over rows
        for j in range(shape[1]):  # Iterate over columns
            # Assign the value from the vector using the integer in the cell as an index
            a =  v[oldMatrix[i, j],0]
            matrix[i, j] = a
    return matrix
def testRunAge():
    probDeath = np.array([0,0.01, 0.02])
    transition_matrix = np.array([[0.028, 0.65, 0.322 ],[0.1, 0.5, 0.4],[0.1, 0.5, 0.4]])
    transition_matrix = generate_transition_matrixAgeDistribution(transition_matrix, 3, probDeath,1)
    # need to apply to each sub frame
    c = MC.precompute_cumulative_probabilities(transition_matrix)
  
    num_chains = 100
    triple = 0.1,10,num_chains
    l =  len(transition_matrix)
    a = generate_reward_vector(l)
    b = generate_utility_vector(l)
    c = generate_age_vector(num_chains)
    t = transition_matrix
    states = simulationwithAge(t, a, b, c, triple)
    return states

def checkTypesAges(a,b,c,factor):
    l = len(a)
    a1 = type(a)
    a2 = type(b)
    a3 = type(c)
    c1 = MC.generate_transition_matrix(l)
    c2 = MC.generate_reward_vector(l)
    c3 = MC.generate_utility_vector(l)
    b1 = type(c1)
    b2 = type(c2)
    b3 = type(c3)

    a_list = list(c1.shape)
    a_list[1] = c1.shape[1]*factor
    c1 = tuple(a_list)

    if a1 == b1 and a2 == b2 and a3 == b3 and a2 == a3:
        print("TYPE SUCCESS")
    if a.shape ==  c1 and c2.shape == b.shape and c3.shape == c.shape:
        print("DIMENSION SUCCCESS")

    else:
        print("TYPE ERROR OR DIMENSION ERROR")
        print(a.shape)
        print(c1)
        print(c2.shape)
        print(b.shape)
        print(c3.shape)
        print(c.shape)
        sys.exit()
    return None 


#########   COMPLETE RUNS ##########


def realLifeSimulation(transition_matrix, rewardVector, utilityVector, *triple):
    if not check_same_size(rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
    (discountRate, num_steps, num_chains) = triple
    initial_state = 0  # Ensure initial state is appropriate
    stop_state = 28 # This is a terminal state
    base = 1 + discountRate
   
    # BASE MATRICES
    discountMatrix = compute_powers(base, (0,num_steps-1))
    discountMatrix = 1/discountMatrix
    cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)

    # RANDOM_NUMBERS
    states = simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state)
    oldStates = copy.deepcopy(states)
    oldStates2 = copy.deepcopy(oldStates)

    # REWARDS
    rewards = populate_matrix_with_indices(rewardVector, states)
    rewards = rewards @ discountMatrix

    # UTILITIES
    integerUtility = 100*utilityVector  #  as dtype of states is int64
    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
    return oldStates2



    if not check_same_size(transition_matrix, rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
    size = 20 # size is supposed to be number of different states in transition matrix {0,1,2, ..., 20} allowable states
    initial_state = 0  # Ensure initial state is appropriate
    stop_state = 28 # This is a terminal state
    base = 1 + discountRate

    num_steps =  50 #this is the number of simulation steps, this is what we will see in output for paths
    num_chains = 10**6 #each chain represents a different patient journey, here we have 100,000 patients
   
    # BASE MATRICES
    discountMatrix = compute_powers(base, (0,num_steps-1))
    discountMatrix = 1/discountMatrix

    cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)

    # RANDOM_NUMBERS
    states = simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state)
    
    oldStates = copy.deepcopy(states)

    rewards = populate_matrix_with_indices(rewardVector, states)
    rewards = rewards @ discountMatrix

    integerUtility = 100*utilityVector  #  as dtype of states is int64
    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
    return rewards, utilities
# gere we actually fix the state matrix to check calculations this is a key test 
def test_initialisation():

    # BASE PARAMETERS
    size = 4 # size is supposed to be number of different states in transition matrix {0,1,2,3} allowable states
    initial_state = 0  # Ensure initial states are appropriate
    stop_state =3
    discountRate = 0.1
    base = 1 + discountRate

    num_steps =  5 #this is the number of simulation steps, this is what we will see in output for paths
    num_chains = 1
   
    # BASE MATRICES
    transition_matrix = generate_transition_matrix(size) #this has twenty states
    rewardVector = np.matrix([[5],[7], [11], [14]])
    utilityVector = np.matrix([[0.5],[0.2], [0.6], [0.1]])
    discountMatrix = compute_powers(base, (0,num_steps-1))
    discountMatrix = 1/discountMatrix
    cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)


    # RANDOM_NUMBERS
    """states = simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state)
    """
    states =  np.matrix([[0,1, 2, 2,2]]) 
    oldStates = copy.deepcopy(states)

    # FLATTENING
    """ print(states.shape)
    print(rewardVector.shape)
    print(discountMatrix.shape)
    print("Hi")
    print(f" The states are :{states}")"""

    rewards = populate_matrix_with_indices(rewardVector, states)

  
    
   
    rewards = rewards @ discountMatrix

    """print(f" The states are now: {states}")
   
    print(f" The discountMatrix is: {discountMatrix}")
    print(f" The rewards are now: {rewards}")

    print("Next")
    print(f" The discountMatrix is: {discountMatrix}")
    print(f" The utilityVector is now: {utilityVector}")
    print(f" The old states are now: {oldStates}")"""


    integerUtility = 100*utilityVector  #  as dtype of states is int64
    integerUtility = integerUtility.astype(int)


    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
   
  
    """print(f" The utilities are now: {utilities}")"""

    if round(rewards[0,0], 8) == 36.23215627 and round(utilities[0,0], 8) == 2.0382829:
        print("PASSED TEST, CALCULATIONS ARE SOUND!")
    else:
        print("NOT PASSED TEST")

    return None

    for x in range(low, high+1):
        num_chains = 10**x
        cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)
     
        start_time = time.time()
        result_with_cache = simulate_markov_with_collapse(transition_matrix, cumulative_probs_cache,initial_state, num_steps, num_chains, stop_state, rewardVector, utilityVector, discountRate)

        end_time = time.time()
        print(f"Time to simulate with: {num_chains} chains: ", end_time - start_time)
    return None


    """start_time = time.time()"""
    for x in range(0, number):
        num_chains = 10**5
        cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)
        
        
        result_with_cache = simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache,initial_state, num_steps, num_chains, stop_state)
        results_with_cache = ""
        print(x)
    """end_time = time.time()
    print(f"Time to simulate with: {number} simulations, with 10^5 chains: ", end_time - start_time)"""
    return None

# take the states matrix and fill it with populate_matrix_numba with values, then dot product with the discount vector
# this is actually a realistic scenario run
def mockSimulation():

   # BASE PARAMETERS
    size = 7 # size is supposed to be number of different states in transition matrix {0,1,2, ..., 7} allowable states
    initial_state = 0  # Ensure initial state is appropriate
    stop_state =  5 #This is a terminal state
    discountRate = 0.1 # this is the annual discount rate
    base = 1 + discountRate

    num_steps =  50 #this is the number of simulation steps, this is what we will see in output for paths
    num_chains = 10**6 #each chain represents a different patient journey, here we have 100,000 patients
   
    # BASE MATRICES
    transition_matrix = generate_transition_matrix(size) #this has twenty states
    rewardVector = generate_reward_vector(size)
    utilityVector = generate_utility_vector(size)
    discountMatrix = compute_powers(base, (0,num_steps-1))
    discountMatrix = 1/discountMatrix

    cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)

    # RANDOM_NUMBERS
    states = simulate_markov_chain_with_cache(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state)
    oldStates = copy.deepcopy(states)
    originalStates = copy.deepcopy(oldStates)

    rewards = populate_matrix_with_indices(rewardVector, states)
    rewards = rewards @ discountMatrix

    integerUtility = 100*utilityVector  #  as dtype of states is int64
    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
    return originalStates

#transition matrix is 7 x 581
#cumulative prob cache should also be 7 x 581
def simulationwithAge(transition_matrix, rewardVector, utilityVector,ageVector, *triple):
    if not check_same_size(rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
    (discountRate, num_steps, num_chains) = triple
    initial_state = 0  # Ensure initial state is appropriate
    stop_state = 5 # This is a terminal state
    base = 1 + discountRate
   
    # BASE MATRICES
    discountMatrix = compute_powers(base, (0,num_steps-1))
    discountMatrix = 1/discountMatrix
    cumulative_probs_cache = precompute_cumulative_probabilitieswithAge(transition_matrix)
   
    # RANDOM_NUMBERS
    quad = (initial_state, num_steps, num_chains, stop_state)
    states = simulate_with_cache_ageDistribution(cumulative_probs_cache, ageVector, *quad)
    oldStates = copy.deepcopy(states)
    oldStates2 = copy.deepcopy(oldStates)

    print(np.unique(oldStates))
    

    # REWARDS
    rewards = populate_matrix_with_indices(rewardVector, states)
    rewards = rewards @ discountMatrix

    # UTILITIES
    integerUtility = 100*utilityVector  #  as dtype of states is int64
    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
    return oldStates2



#########   TRNASITION MATRICES WITH AGE ##########


#TESTED
# probDeath is a num_steps lenght vector for additional probability of death at age x. ageDeath is for MALSD patients, starts at 18. 
# here actually format must follow Input_v3 posDeath is the death position index column we wish to intervene on.
# the new transition matrix is now #states x (#states*#num_steps) with each #states x #states block representing a transition
# matrix for a patient of age = age so for patient at step =k, we query transition[current_state + #states*age]
# to get the vector of size #states of transition to next states, and same with cumulative prob cache. First age is 18.
#TESTED
def generate_transition_matrixAgeDistribution(transition_matrix, size, probDeath,posDeath):
    num_ages =probDeath.shape[0]    
    matrices = [transition_matrix] *num_ages
    # Horizontally stack the matrices
    matrix = np.hstack(matrices)
  
    # MASLD should be first row 
    for x in range(0, num_ages):
            matrix[0,posDeath + x*size] = transition_matrix[0,posDeath] + probDeath[x]    # increment death node
            matrix[0,x*size] = 1- sum(matrix[0,x*size +1: (x+1)*(size)])  #increment MASLD prob
    return matrix







