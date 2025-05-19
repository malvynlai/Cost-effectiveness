
def realLifeSimulationold2(transition_matrix, rewardVector, utilityVector, discountRate, num_steps, num_chains):
    if not check_same_size(transition_matrix, rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
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
    return rewards, utilities

def realLifeSimulationOld(transition_matrix, rewardVector, utilityVector, discountRate):

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

# this is actually not more efficient

def utilityComputationStatic(path, rewardVector, utilityVector, discountRate):
    totalReturn = 0
    totalUtility = 0 
    i = 0
    
    for x in path:
        totalReturn += rewardVector[x]/(1+discountRate)**i
        totalUtility += utilityVector[x]/(1+discountRate)**i
        i +=1
    return (int(totalReturn),int(totalUtility),i+1)
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
# probDeath is a num_steps lenght vector for additional probability of death at age x. ageDeath is for MALSD patients, starts at 18. 
# here actually format must follow Input_v3



def realLifeSimulationold2(transition_matrix, rewardVector, utilityVector, discountRate, num_steps, num_chains):
    if not check_same_size(transition_matrix, rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
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
    return rewards, utilities
def realLifeSimulationold2(transition_matrix, rewardVector, utilityVector, discountRate, num_steps, num_chains):
    if not check_same_size(transition_matrix, rewardVector, utilityVector):
        print("Input size error!")
        sys.exit()

   # BASE PARAMETERS
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
    return rewards, utilities
# HCC distribution knowing early and late is enough to derive intermediate
# early and late should be percentages
# HCC transition is transition from HCC to next states[early transitions], [intermediate transitions], [late transitions]
# This forms the basis of more simulations as we sensitise the early-late axis
def HCCdistribution(early, late, HCC_index, rewardVector, transition_matrix, utilityVector, HCCreward, HCCtransition, HCCutility):
    intermediate  = 1 - early -late
    distribution = np.array(early, intermediate, late)
    rewardVector[HCC_index] = np.dot(distribution, HCCreward)
    utilityVector[HCC_index] = np.dot(distribution, HCCutility)
    transition_matrix[HCC_index, :] = np.dot(distribution, HCCtransition)
    return transition_matrix, utilityVector, rewardVector
    
# above needs to be tested

def vectorSum(rewards, utilities):

    reward = np.sum(rewards)
    utility = np.sum(utilities)
    return reward, utility
