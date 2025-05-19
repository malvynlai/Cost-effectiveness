import numpy as np
from numba import njit
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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

def generate_death_rate(states):
    # Returns a python list containing the death rate for each respective state (with respect to time)
    # Arbitrarily define patient's state=17 as death
    death_rates = []
    for state in states.T: 
        death_num = np.count_nonzero(state == 17)
        patients_num = state.size
        death_rate = death_num / patients_num
        death_rates.append(death_rate)
    return death_rates

def visualize_death_rates(death_rates):
    num_steps = len(death_rates)
    x_axis = list(range(num_steps))
    plt.plot(x_axis, death_rates, marker='o', linestyle='-', color='b')
    plt.xticks(np.arange(0, num_steps, 1))
    plt.xlabel('Step Number')
    plt.ylabel('Death Rate')
    plt.title('Death Rate by Step')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.show()
@njit 
def compute_powers(base, exponent_range):
    # Create an array of exponents
    exponents = np.arange(exponent_range[0], exponent_range[1] + 1)
    
    # Compute the powers using broadcasting
    powers = base ** exponents
    
    return powers

# take the states matrix and fill it with populate_matrix_numba with values, then dot product with the discount vector
# this is actually a realistic scenario run
def mock_initialisation():

   # BASE PARAMETERS
    size = 20 # size is supposed to be number of different states in transition matrix {0,1,2, ..., 20} allowable states
    initial_state = 0  # Ensure initial state is appropriate
    stop_state = 28 # This is a terminal state
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
    death_rates = generate_death_rate(states)
    visualize_death_rates(death_rates)
    oldStates = copy.deepcopy(states)

    rewards = populate_matrix_with_indices(rewardVector, states)
    rewards = rewards @ discountMatrix

    integerUtility = 100*utilityVector  #  as dtype of states is int64
    utilities =  populate_matrix_with_indices(integerUtility, oldStates)
    utilities = utilities @ discountMatrix
    utilities = utilities/100
    return None

def check_same_size(*arrays):
    # Get the shape or size of the first array
    first_shape = arrays[0].shape if len(arrays[0].shape) > 1 else (len(arrays[0]),)
    
    # Check if all other arrays (matrices or vectors) have the same size
    for array in arrays[1:]:
        current_shape = array.shape if len(array.shape) > 1 else (len(array),)
        if current_shape != first_shape:
            return False  # If any array is not the same size, return False
    return True 

def realLifeSimulation(transition_matrix, rewardVector, utilityVector, discountRate):

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

def getInputs(path, sheet_name):
    df = pd.read_excel(path, sheet_name= sheet_name)
    columns = list(df.iloc[0,1:-1])
    df2 = pd.DataFrame(index = columns, columns = columns)
    for x in range(0, len(columns)):
        for y in range(0, len(columns)):
            df2.iloc[x,y] = df.iloc[x+1,y+1]
    print(df2)
    return df2

def getUtilities(path, sheet_name):
    return pd.read_excel(path, sheet_name= sheet_name,index_col=0)

def check_if_array(matrices):
    # Check if each element in the list is a NumPy array
    return all(isinstance(matrix, np.ndarray) for matrix in matrices)

# Returns if a control vs 
def costEffectiveness(costControl, costIntervention, lifeControl, lifeIntervention, threshold):
    costIncrement =  costIntervention- costControl
    lifeIncrement = lifeIntervention - lifeControl
    value = costIncrement/lifeIncrement
    if value> threshold:
        return (value, False)
    else: 
        return (value, True)

def gatheringInputs(file_path):
    sheet_name = "p4"
    df = getInputs(file_path,sheet_name)
    matrix = df.values

    sheet_name = "States"
    d = getUtilities(file_path,sheet_name)
    rewardVector = d.iloc[1,:].to_numpy(dtype=int)
    utilityVector = d.iloc[0,:].to_numpy(dtype=float)

    if not (check_same_size(matrix, rewardVector, utilityVector) and check_if_array(matrix, rewardVector, utilityVector)):
        sys.exit()
    else:
        return matrix,rewardVector, utilityVector


# Screening effectiveness

if __name__ ==  "__main__":
    
    #file_path = r"C:\Users\sovan\Box\Sovann Linden's Files\Cost-effectiveness\Inputs_CEA.xlsx"  # Replace with your file path
    #matrix,rewardVector,utilityVector = gatheringInputs(file_path)
    #lifeControl, costControl = realLifeSimulation(transition_matrix, rewardVector, utilityVector)
    




    """

    matrix = np.matrix([[1, 1], [0, 0]],dtype=np.int64)
    vector = np.matrix([[5],[7]],dtype=np.int64)
    num_chains = 10**6
    
    populate_matrix_with_indices(vector, matrix)
    um_chains = 10**6
    """
    num_chains = 10**4
    test_initialisation()
    start_time = time.time()
    mock_initialisation()
    print(f"Time to simulate with: {num_chains} chains: ", time.time() - start_time)


#For next time: fix speed on populate_matrix with indices