import numpy as np
from numba import njit
import sys
import time

  # For sizing num_states: 20, num_chains: high, num_steps: 50
  # initial state here is common for all people
  # stop state is also common here for all people
import warnings
import copy
warnings.filterwarnings("ignore")

@njit
def precompute_cumulative_probabilities(transition_matrix):
    num_states = transition_matrix.shape[0]
    cumulative_probs_cache = np.zeros((num_states, num_states))
    for s in range(num_states):
        cumulative_probs_cache[s, 0] = transition_matrix[s, 0]
        for j in range(1, num_states):
            cumulative_probs_cache[s, j] = cumulative_probs_cache[s, j - 1] + transition_matrix[s, j]
    return cumulative_probs_cache

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

# this is actually not more efficient
import numpy as np
@njit
def simulate_markov_with_collapse(transition_matrix, cumulative_probs_cache, initial_state, num_steps, num_chains, stop_state, rewardVector, utilityVector, discountRate):
    num_states = transition_matrix.shape[0]
    
    # Initialize vectors for total returns, utilities, and lifetimes
    totalReturnVector = np.zeros(num_chains)
    totalUtilityVector = np.zeros(num_chains)
    totalLifetimeVector = np.zeros(num_chains)

    # Generate all the random numbers we need
    random_numbers = np.random.rand(num_chains, num_steps - 1)

    # Iterate through time steps
    for t in range(num_steps):
        for i in range(num_chains):
            # Set initial state for each chain
            if t == 0:
                current_state = initial_state
            else:
                # Skip if already at stop state
                if totalLifetimeVector[i] > 0 and current_state == stop_state:
                    continue
            
            cumulative_probs = cumulative_probs_cache[current_state]
            # Convert a random draw into a state transition
            current_state = int(np.searchsorted(cumulative_probs, random_numbers[i, t - 1]))
            
             #a = rewardVector[0] / (discountRate ** (t + 1))
             #totalReturnVector[i] =  totalReturnVector[i] + a

            # Accumulate returns and utilities
            #totalReturnVector[i] += rewardVector[0] / (discountRate ** (t + 1))
            #totalUtilityVector[i] += utilityVector[0] / (discountRate ** (t + 1))

            # Update the lifetime vector
            totalLifetimeVector[i] = t + 1  # +1 to count the current step

            # Check for the stop state
            if current_state == stop_state:
                break

    # Convert to int for return values
    #totalReturnVector = totalReturnVector.astype(int)
    #totalUtilityVector = totalUtilityVector.astype(int)

    return totalReturnVector, totalUtilityVector, totalLifetimeVector

def utilityComputationStatic(path, rewardVector, utilityVector, discountRate):
    totalReturn = 0
    totalUtility = 0 
    i = 0
    
    for x in path:
        totalReturn += rewardVector[x]/(1+discountRate)**i
        totalUtility += utilityVector[x]/(1+discountRate)**i
        i +=1
    return (int(totalReturn),int(totalUtility),i+1)

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

@njit 
def compute_powers(base, exponent_range):
    # Create an array of exponents
    exponents = np.arange(exponent_range[0], exponent_range[1] + 1)
    
    # Compute the powers using broadcasting
    powers = base ** exponents
    
    return powers

@njit
def populate_matrix_with_indices(v, matrix):
    shape = matrix.shape
    oldMatrix = matrix #as we change value of matrix
    
    for i in range(shape[0]):  # Iterate over rows
        for j in range(shape[1]):  # Iterate over columns
            # Assign the value from the vector using the integer in the cell as an index
        
            matrix[i, j] = v[oldMatrix[i, j]]

    
    return matrix

# take the states matrix and fill it with populate_matrix_numba with values, then dot product with the discount vector

def mock_initialisation():

   # BASE PARAMETERS
    size = 20 # size is supposed to be number of different states in transition matrix {0,1,2,3} allowable states
    initial_state = 0  # Ensure initial states are appropriate
    stop_state =28
    discountRate = 0.1
    base = 1 + discountRate

    num_steps =  50 #this is the number of simulation steps, this is what we will see in output for paths
    num_chains = 10**5
   
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
    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
   

    return None

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


    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
   
  
    """print(f" The utilities are now: {utilities}")"""

    if round(rewards[0,0], 8) == 36.23215627 and round(utilities[0,0], 8) == 2.0382829:
        print("PASSED TEST, CALCULATIONS ARE SOUND!")
    else:
        print("NOT PASSED TEST")

    return None


def timingMonster(low, high):
    for x in range(low, high+1):
        num_chains = 10**x
        cumulative_probs_cache = precompute_cumulative_probabilities(transition_matrix)
     
        start_time = time.time()
        result_with_cache = simulate_markov_with_collapse(transition_matrix, cumulative_probs_cache,initial_state, num_steps, num_chains, stop_state, rewardVector, utilityVector, discountRate)

        end_time = time.time()
        print(f"Time to simulate with: {num_chains} chains: ", end_time - start_time)
    return None

@njit
def timingDifferentStructures(number):
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

if __name__ ==  "__main__":
    matrix = np.matrix([[1, 1], [0, 0]],dtype=np.int64)
    vector = np.matrix([[5],[7]],dtype=np.int64)
    num_chains = 10**4
    start_time = time.time()
    populate_matrix_with_indices(matrix, vector)
    """mock_initialisation()
    print(f"Time to simulate with: {num_chains} chains: ", time.time() - start_time)"""

#For next time: fix speed on populate_matrix with indices