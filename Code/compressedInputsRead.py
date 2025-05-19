import pandas as pd
import numpy as np
import sys


#########  SECTIONS OVERVIEW ######### 


#  Overall flow: return transition, ageVector,costutility, probdeath
#  1. Read data from excel file INPUT_v3, but can also generate objects here (if want to override some inputs for simulation purposes)
#  2. generateC
#  3. No simulation logic/random numbers in this file, refer to MCMC_improved_cleanv3.py
#  4. This is not an execution file, call completeSimulationRunv2.py


#########  SECTIONS OVERVIEW ######### 

# Simply reads on ReadingSheet the transition matrix from excel
def generateCleanTransition(df):
    df = df.iloc[0:7, 1:8]
    return df

# Collects data for different HCC grades as organised in Excel sheet
def returnSpacedArray(index, column,df):
    return np.array([df.iloc[index +3*x,column] for x in [0,1,2]])

# Rebuilds transition matrix (automates excel formulas) from elementary inputs
# HCC is HCC distribution 3x1 vector
# input_dict comprises other elementary inputs
# Use this if you want to generate many different transition matrices for each elementary input combination (sensitivity analysis)
def generateFinalTransition(df, input_dict, HCC):
    cUDRate = input_dict['cirrhosisUnderdiagnosisRateInMasld_Rate']
    cUDHCC = input_dict['cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD']
    cHCC = input_dict['cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic']
    falsePositiveHCC_masld  = input_dict['masldIncidenceRates_falsePositiveHCC']
    cirrhosis_masld= input_dict['masldIncidenceRates_masldToCirrhosis']

    for x in range(0,1):
        masld_falsePositiveHCC = falsePositiveHCC_masld
        masld_hcc, masld_death, masld_cirrhosis =df.iloc[1,1], df.iloc[5,1], df.iloc[6,1]    
        hcc_falsePositiveHCC, hcc_death, hcc_cirrhosis= df.iloc[4,2],df.iloc[5,2],df.iloc[6,2]
        treatment_treated = 1
        treatment_death,  hcc_treatment = df.iloc[5,3],df.iloc[2,2]

    death_masld = cUDRate* df.iloc[3,13]  + (1-cUDRate)*df.iloc[4,13]
    hcc_masld = cUDRate*cUDHCC+ (1 - cUDRate)*cHCC    
    masld_masld = 1 - hcc_masld - death_masld - falsePositiveHCC_masld - cirrhosis_masld
    
    death_treated = np.dot(returnSpacedArray(1,22,df), HCC)
    treated_treated = np.dot(returnSpacedArray(0,22,df), HCC)
    recurrence_treated = 1 -death_treated - treated_treated

    treatment_hcc = np.dot(returnSpacedArray(0,17,df), HCC)
    death_hcc =np.dot(returnSpacedArray(1,17,df), HCC)
    treatedOutcomesRates_recurrence = 1- treatment_hcc - death_hcc
    hcc_treated = np.dot(HCC, treatedOutcomesRates_recurrence)
    hcc_hcc = 1 - treatment_hcc - death_hcc 
  
    treated_death = df.iloc[5,4]
    data = {
        "MASLD": [masld_masld, hcc_masld, 0, 0, masld_falsePositiveHCC, death_masld, cirrhosis_masld],
        "HCC": [0, hcc_hcc, treatment_hcc, 0, 0, death_hcc, 0],
        "Treatment": [0, 0, 0, 1, 0, 0, 0],
        "Treated": [0,0,0,treated_treated,0, death_treated, 0],
        "False Positive HCC": [1,0,0,0,0,0,0],
        "Death": [0,0,0,0,0,1,0],
        "Cirrhosis": [0,0,0,0,0,0,1]
    }
    c = ["MASLD", "HCC", "Treatment", "Treated", "False Positive HCC", "Death", "Cirrhosis"]
    transition_matrix = pd.DataFrame(data, index = c).transpose()
    return transition_matrix

# Reads from excel vanilla cost utility matrices
def generate_cost_utility_matrix(df):
    data = {
        'Control Cost': [df.iloc[0, 30], df.iloc[1, 30], df.iloc[2, 30], df.iloc[3, 30], df.iloc[4, 30], df.iloc[5, 30], df.iloc[6, 30]],
        'Control Utility': [df.iloc[0, 32], df.iloc[1, 32], df.iloc[2, 32], df.iloc[3, 32], df.iloc[4, 32], df.iloc[5, 32], df.iloc[6, 32]],
    }

    index = ['MASLD', 'HCC', 'Treated', 'Untreated', 'False Positive HCC', 'Death', 'Cirrhosis']
    cost_utility_matrix = pd.DataFrame(data, index=index)
    return cost_utility_matrix

# must be of length 83, probabilities of MASLD patients of dying at that age naturally
def generate_prob_death(df):
    probDeath = df.iloc[:100-18+1, 2]
    return probDeath.to_numpy()

# Normalises age vector with indices from the proportions given for each age group.
# Generates a num_chains x 1 vector of ages.
def generate_age_vector(df, num_chains):
    population_size = num_chains
    
    proportions = df['Proportion'].values
    proportions = proportions / proportions.sum()
    
    counts = (proportions * population_size).astype(int)
    
    remaining = population_size - counts.sum()
    if remaining > 0:
        decimal_parts = (proportions * population_size) - counts
        indices = np.argsort(decimal_parts)[-remaining:]
        counts[indices] += 1
    
    age_vector = np.repeat(df.index.values, counts)
    assert len(age_vector) == population_size, f"Expected {population_size} samples, got {len(age_vector)}"
    
    return age_vector




