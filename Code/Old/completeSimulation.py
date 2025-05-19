import compressedInputsRead as readInputs
import MCMC_improved_clean as MC
import numpy as np
import pandas as pd
import sys
import time
import Visualizations.vizFinal as viz

def checkTypes(a,b,c):
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
    if a1 == b1 and a2 == b2 and a3 == b3 and a2 == a3:
        print("TYPE SUCCESS")
    if a.shape ==  c1.shape and c2.shape == b.shape and c3.shape == c.shape:
        print("DIMENSION SUCCCESS")
    else:
        print("TYPE ERROR OR DIMENSION ERROR")
        sys.exit()
    return None 
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
def completeRun(file, input_dict, HCC):
    df = pd.read_excel(file, sheet_name='ReadingSheet')  
    discountRate = 0.1
    num_steps = 10
    num_chains = 10**2

    t = readInputs.generateFinalTransition(df, input_dict, HCC).to_numpy()
    t[0,1] = 0
    t[0,4] =0
    t[0,6] = 0 
    t[0,0] = 0.9
    t[0,5] = 0.1
    print(t)

    costUtility = readInputs.generate_cost_utility_matrix(df)
    rewardVector = np.expand_dims(costUtility.iloc[:,0].to_numpy(), axis = 1)
    utilityVector = np.expand_dims(costUtility.iloc[:,1].to_numpy(), axis = 1)

    checkTypes(t,rewardVector,utilityVector)
    triple =  (discountRate, num_steps, num_chains)
    states = MC.realLifeSimulation(t, rewardVector, utilityVector, *triple)
    return states

def completeRunAge(file, input_dict, HCC):
    df = pd.read_excel(file, sheet_name='ReadingSheet')  
    discountRate = 0.1
    num_steps = 50
    num_chains = 10**6
    triple =  (discountRate, num_steps, num_chains)
    posDeath = 5
    ageRange = 83

    df1 = pd.read_excel(file, sheet_name='ActuarialTables')  
    probDeath = readInputs.generate_prob_death(df1)
    df2 = pd.read_excel(file, sheet_name='AgeVector')  
    ageVector = readInputs.generate_age_vector(df2,num_chains)

    t = readInputs.generateFinalTransition(df, input_dict, HCC).to_numpy()
    sizeAges = len(t)

    t = MC.generate_transition_matrixAgeDistribution(t, sizeAges, probDeath,posDeath)

    costUtility = readInputs.generate_cost_utility_matrix(df)
    rewardVector = np.expand_dims(costUtility.iloc[:,0].to_numpy(), axis = 1)
    utilityVector = np.expand_dims(costUtility.iloc[:,1].to_numpy(), axis = 1)

    checkTypesAges(t,rewardVector,utilityVector,ageRange)
    t1 = time.time()

    states = MC.simulationwithAge(t, rewardVector, utilityVector, ageVector, *triple)
    print(time.time()-t1)
    return states

file = r"C:\Users\joannekl\Box\Cost-effectiveness\Inputs\Inputs_CEA_v3_t1.xlsx"
input_dict = {
        'cirrhosisUnderdiagnosisRateInMasld_Rate': 0.059,
        'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD': 0.043,
        'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic': 0.0011,
        'masldIncidenceRates_falsePositiveHCC': 0.1,
        'masldIncidenceRates_masldToCirrhosis': 0.006}
HCC = np.array([0.457,0.23,0.313])


s = completeRun(file, input_dict, HCC)
s = pd.DataFrame(s)
s.to_csv(r"C:\Users\joannekl\Box\Cost-effectiveness\Inputs\out.csv")
viz.graphs(s)

print(s)


