from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import os
import uuid
from tqdm import tqdm
from Visualizations.sensFinal import two_way_plot
import multiprocessing
from completeSimulationv2_ver import completeRunAge
from shutil import copyfile
import math
import copy

# Create output directories if they don't exist
output_base = "/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Data"
graphs_dir = "/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Graphs"
os.makedirs(output_base, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

original_file = "/sailhome/malvlai/Cost-effectiveness/Inputs/Inputs_CEA_v4_3.27.25.xlsx"
output_base = "/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Data"

input_dict = {
    'cirrhosisUnderdiagnosisRateInMasld_Rate': 0.059,
    'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD': 0.043,
    'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic': 0.0011,
    'masldIncidenceRates_falsePositiveHCC': 0.1,
    'masldIncidenceRates_masldToCirrhosis': 0.006
}

HCC = np.array([0.457, 0.23, 0.313])


def average_hcc_and_five_year_survival(states):
    n_patients = states.shape[0]
    has_hcc = np.zeros(n_patients, dtype=np.int32)
    hcc_age = np.zeros(n_patients, dtype=np.int32)
    survived = np.zeros(n_patients, dtype=np.int32)
    
    for i in range(n_patients):  
        for j in range(states.shape[1]):
            if states[i,j] == 2 or states[i,j] == 3:
                has_hcc[i] = 1
                survived[i] = 1  
                for k in range(5):
                    if j + k < states.shape[1] and states[i, j + k] == 5:
                        survived[i] = 0  
                        break
                break
    hcc_ages = hcc_age[has_hcc == 1]
    survival_rate = np.sum(survived[has_hcc == 1]) / np.sum(has_hcc) if np.sum(has_hcc) > 0 else 0
    
    return np.mean(hcc_ages) + 18 if len(hcc_ages) > 0 else 0, survival_rate


def average_death_from_hcc(states):
    n_patients = states.shape[0]
    death_times = np.zeros(n_patients, dtype=np.int32)
    has_hcc_death = np.zeros(n_patients, dtype=np.int32)
    
    for i in range(n_patients):
        row = states[i] 
        hcc_indices = np.where((row == 2) | (row == 3))[0]
        if len(hcc_indices) > 0:
            hcc_start = hcc_indices[0]
            if 5 in row:  # If death occurs
                death_idx = np.where(row == 5)[0][0]
                death_times[i] = death_idx - hcc_start
                has_hcc_death[i] = 1
    
    valid_times = death_times[has_hcc_death == 1]
    if len(valid_times) == 0:
        return 0, 0
    
    return np.mean(valid_times), np.median(valid_times)


# 0 - 6, 1 - 7, (-2, -1) ie B12 == 10, 1
def build_reading_sheet(sheets):
    final_transition = sheets['FinalTransition-Control']
    final_rewards = sheets['FinalRewards']
    intervention_transition = sheets['FinalTransition-Intervention']
    df = sheets['ReadingSheet']

    # Downstream equations for control transition matrix
    final_transition.iloc[10, 1] = final_transition.iloc[16, 1] * final_transition.iloc[17, 1] + (1 - final_transition.iloc[16, 1]) * final_transition.iloc[18, 1] # Checked, completed
    final_transition.iloc[12, 1] = final_transition.iloc[16, 1] * final_transition.iloc[19, 1] + (1 - final_transition.iloc[16, 1]) * final_transition.iloc[20, 1] # Checked, completed
    # Below 6 eq are for HCC outcome rates
    final_transition.iloc[23, 2] = final_transition.iloc[82, 2] * (1 - final_transition.iloc[16, 1]) + final_transition.iloc[82, 3] * final_transition.iloc[16, 1] if math.isclose(final_transition.iloc[23, 2], .5327) else final_transition.iloc[23, 2] # Checked, completed
    final_transition.iloc[25, 2] = 1 - final_transition.iloc[23, 2] # Checked, completed
    final_transition.iloc[26, 2] = final_transition.iloc[91, 2] * (1 - final_transition.iloc[16, 1]) + final_transition.iloc[91, 3] * final_transition.iloc[16, 1] if math.isclose(final_transition.iloc[26, 2], .5332) else final_transition.iloc[26, 2] # Checked, completed
    final_transition.iloc[28, 2] = 1 - final_transition.iloc[26, 2] # Checked, completed
    final_transition.iloc[29, 2] = final_transition.iloc[98, 2] * (1 - final_transition.iloc[16, 1]) + final_transition.iloc[98, 3] * final_transition.iloc[16, 1] if math.isclose(final_transition.iloc[29, 2], .3886) else final_transition.iloc[29, 2] # Checked, completed
    final_transition.iloc[31, 2] = 1 - final_transition.iloc[29, 2] # Checked, completed
    final_transition.iloc[32, 2] = final_transition.iloc[23, 2] * final_transition.iloc[63, 1] + final_transition.iloc[26, 2] * final_transition.iloc[64, 1] + final_transition.iloc[29, 2] * final_transition.iloc[65, 1] # Checked, completed
    
    # Control transition matrix (MASLD + HCC)
    df.iloc[0, 2] = final_transition.iloc[10, 1] # Checked, completed 
    df.iloc[0, 5] = final_transition.iloc[11, 1] * final_transition.iloc[63, 4] # Checked, completed
    df.iloc[0, 6] = final_transition.iloc[12, 1] # Checked, completed
    df.iloc[0, 7] = final_transition.iloc[13, 1] # Checked, completed
    df.iloc[0, 1] = 1 - df.iloc[0, 2] - df.iloc[0, 5] - df.iloc[0, 6] - df.iloc[0, 7] # Checked, completed
    df.iloc[1, 4] = final_transition.iloc[32, 2] # Checked, completed
    df.iloc[1, 3] = 1 - df.iloc[1, 4] # Checked, completed
    # Downstream equations for control matrix 
    final_transition.iloc[69, 1] = (final_transition.iloc[25, 2] * final_transition.iloc[63, 1]) / (final_transition.iloc[25, 2] * final_transition.iloc[63, 1] + final_transition.iloc[28, 2] * final_transition.iloc[64, 1] + final_transition.iloc[31, 2] * final_transition.iloc[65, 1]) # Checked, completed
    final_transition.iloc[70, 1] = (final_transition.iloc[28, 2] * final_transition.iloc[64, 1]) / (final_transition.iloc[25, 2] * final_transition.iloc[63, 1] + final_transition.iloc[28, 2] * final_transition.iloc[64, 1] + final_transition.iloc[31, 2] * final_transition.iloc[65, 1]) # Checked, completed
    final_transition.iloc[71, 1] = (final_transition.iloc[31, 2] * final_transition.iloc[65, 1]) / (final_transition.iloc[25, 2] * final_transition.iloc[63, 1] + final_transition.iloc[28, 2] * final_transition.iloc[64, 1] + final_transition.iloc[31, 2] * final_transition.iloc[65, 1]) # Checked, completed
    final_transition.iloc[66, 1] = (final_transition.iloc[23, 2] * final_transition.iloc[63, 1]) / (final_transition.iloc[23, 2] * final_transition.iloc[63, 1] + final_transition.iloc[26, 2] * final_transition.iloc[64, 1] + final_transition.iloc[29, 2] * final_transition.iloc[65, 1]) # Checked, completed
    final_transition.iloc[67, 1] = (final_transition.iloc[26, 2] * final_transition.iloc[64, 1]) / (final_transition.iloc[23, 2] * final_transition.iloc[63, 1] + final_transition.iloc[26, 2] * final_transition.iloc[64, 1] + final_transition.iloc[29, 2] * final_transition.iloc[65, 1]) # Checked, completed
    final_transition.iloc[68, 1] = (final_transition.iloc[29, 2] * final_transition.iloc[65, 1]) / (final_transition.iloc[23, 2] * final_transition.iloc[63, 1] + final_transition.iloc[26, 2] * final_transition.iloc[64, 1] + final_transition.iloc[29, 2] * final_transition.iloc[65, 1]) # Checked, completed
    # Untreated outcome rates (53, 55, 57) are varied, so we need to make sure they sum to 1
    final_transition.iloc[52, 2] = 1 - final_transition.iloc[53, 2] # Checked, completed
    final_transition.iloc[54, 2] = 1 - final_transition.iloc[55, 2] # Checked, completed
    final_transition.iloc[56, 2] = 1 - final_transition.iloc[57, 2] # Checked, completed
    # Treated outcome rates (39, 42, 45) are varied, so we need to make sure they sum to 1
    final_transition.iloc[38, 2] = 1 - final_transition.iloc[39, 2] # Checked, completed
    final_transition.iloc[41, 2] = 1 - final_transition.iloc[42, 2] # Checked, completed
    final_transition.iloc[44, 2] = 1 - final_transition.iloc[45, 2] # Checked, completed
    # These are the final 4 eq before the transition matrix
    final_transition.iloc[58, 2] = final_transition.iloc[52, 2] * final_transition.iloc[69, 1] + final_transition.iloc[54, 2] * final_transition.iloc[70, 1] + final_transition.iloc[56, 2] * final_transition.iloc[71, 1] # Checked, completed
    final_transition.iloc[59, 2] = final_transition.iloc[53, 2] * final_transition.iloc[69, 1] + final_transition.iloc[55, 2] * final_transition.iloc[70, 1] + final_transition.iloc[57, 2] * final_transition.iloc[71, 1] # Checked, completed
    final_transition.iloc[47, 2] = final_transition.iloc[38, 2] * final_transition.iloc[66, 1] + final_transition.iloc[41, 2] * final_transition.iloc[67, 1] + final_transition.iloc[44, 2] * final_transition.iloc[68, 1] # Checked, completed 
    final_transition.iloc[48, 2] = final_transition.iloc[39, 2] * final_transition.iloc[66, 1] + final_transition.iloc[42, 2] * final_transition.iloc[67, 1] + final_transition.iloc[45, 2] * final_transition.iloc[68, 1] # Checked, completed
    # Treated/Untreated/False Positive
    df.iloc[2, 3] = final_transition.iloc[58, 2] # Checked, completed
    df.iloc[2, 6] = final_transition.iloc[59, 2] # Checked, completed 
    df.iloc[3, 4] = final_transition.iloc[47, 2] # Checked, completed
    df.iloc[3, 6] = final_transition.iloc[48, 2] # Checked, completed
    df.iloc[4, 2] = df.iloc[0, 2] # Checked, completed 
    df.iloc[4, 5] = 0 # Checked, completed
    df.iloc[4, 6] = df.iloc[0, 6] # Checked, completed
    df.iloc[4, 7] = df.iloc[0, 7] # Checked, completed
    df.iloc[4, 1] = 1 - df.iloc[4, 2] - df.iloc[4, 5] - df.iloc[4, 6] - df.iloc[4, 7] # Checked, completed





    # Downstream equation for intervention transition matrix
    final_transition.iloc[63, 2] = final_transition.iloc[63, 3] * final_transition.iloc[63, 4] + final_transition.iloc[63, 1] * (1 - final_transition.iloc[63, 4])
    final_transition.iloc[64, 2] = final_transition.iloc[64, 3] * final_transition.iloc[63, 4] + final_transition.iloc[64, 1] * (1 - final_transition.iloc[63, 4])
    final_transition.iloc[65, 2] = final_transition.iloc[65, 3] * final_transition.iloc[63, 4] + final_transition.iloc[65, 1] * (1 - final_transition.iloc[63, 4])
    final_transition.iloc[32, 3] = final_transition.iloc[23, 2] * final_transition.iloc[63, 2] + final_transition.iloc[26, 2] * final_transition.iloc[64, 2] + final_transition.iloc[29, 2] * final_transition.iloc[65, 2] # Checked, completed
    # Intervention transition matrix - many of them reference final_transition (control) since the intervention has
    # the same row/values
    intervention_transition.iloc[0, 2] = final_transition.iloc[10, 1] # Checked, completed
    intervention_transition.iloc[0, 5] = final_transition.iloc[11, 1] * final_transition.iloc[63, 4] # Checked, completed
    intervention_transition.iloc[0, 6] = final_transition.iloc[12, 1] # Checked, completed
    intervention_transition.iloc[0, 7] = final_transition.iloc[13, 1] # Checked, completed
    intervention_transition.iloc[0, 1] = 1 - intervention_transition.iloc[0, 2] - intervention_transition.iloc[0, 5] - intervention_transition.iloc[0, 6] - intervention_transition.iloc[0, 7] # Checked, completed
    intervention_transition.iloc[1, 4] = final_transition.iloc[32, 3] # Checked, completed
    intervention_transition.iloc[1, 3] = 1 - intervention_transition.iloc[1, 4] # Checked, completed
    # Downstream equations for intervention
    final_transition.iloc[69, 2] = (final_transition.iloc[25, 3] * final_transition.iloc[63, 2]) / (final_transition.iloc[25, 3] * final_transition.iloc[63, 2] + final_transition.iloc[28, 3] * final_transition.iloc[64, 2] + final_transition.iloc[31, 3] * final_transition.iloc[65, 2]) # Checked, completed
    final_transition.iloc[70, 2] = (final_transition.iloc[28, 3] * final_transition.iloc[64, 2]) / (final_transition.iloc[25, 3] * final_transition.iloc[63, 2] + final_transition.iloc[28, 3] * final_transition.iloc[64, 2] + final_transition.iloc[31, 3] * final_transition.iloc[65, 2]) # Checked, completed
    final_transition.iloc[71, 2] = (final_transition.iloc[31, 3] * final_transition.iloc[65, 2]) / (final_transition.iloc[25, 3] * final_transition.iloc[63, 2] + final_transition.iloc[28, 3] * final_transition.iloc[64, 2] + final_transition.iloc[31, 3] * final_transition.iloc[65, 2]) # Checked, completed
    final_transition.iloc[66, 2] = (final_transition.iloc[23, 3] * final_transition.iloc[63, 2]) / (final_transition.iloc[23, 3] * final_transition.iloc[63, 2] + final_transition.iloc[26, 3] * final_transition.iloc[64, 2] + final_transition.iloc[29, 3] * final_transition.iloc[65, 2]) # Checked, completed
    final_transition.iloc[67, 2] = (final_transition.iloc[26, 3] * final_transition.iloc[64, 2]) / (final_transition.iloc[23, 3] * final_transition.iloc[63, 2] + final_transition.iloc[26, 3] * final_transition.iloc[64, 2] + final_transition.iloc[29, 3] * final_transition.iloc[65, 2]) # Checked, completed
    final_transition.iloc[68, 2] = (final_transition.iloc[29, 3] * final_transition.iloc[65, 2]) / (final_transition.iloc[23, 3] * final_transition.iloc[63, 2] + final_transition.iloc[26, 3] * final_transition.iloc[64, 2] + final_transition.iloc[29, 3] * final_transition.iloc[65, 2]) # Checked, completed
    final_transition.iloc[58, 3] = final_transition.iloc[52, 2] * final_transition.iloc[69, 2] + final_transition.iloc[54, 2] * final_transition.iloc[70, 2] + final_transition.iloc[56, 2] * final_transition.iloc[71, 2] # Checked, completed
    final_transition.iloc[59, 3] = final_transition.iloc[53, 2] * final_transition.iloc[69, 2] + final_transition.iloc[55, 2] * final_transition.iloc[70, 2] + final_transition.iloc[57, 2] * final_transition.iloc[71, 2] # Checked, completed
    final_transition.iloc[47, 3] = final_transition.iloc[38, 2] * final_transition.iloc[66, 2] + final_transition.iloc[41, 2] * final_transition.iloc[67, 2] + final_transition.iloc[44, 2] * final_transition.iloc[68, 2] # Checked, completed
    final_transition.iloc[48, 3] = final_transition.iloc[39, 2] * final_transition.iloc[66, 2] + final_transition.iloc[42, 2] * final_transition.iloc[67, 2] + final_transition.iloc[45, 2] * final_transition.iloc[68, 2] # Checked, completed
    # Treated/Untreated/False Positive (Intervention)
    intervention_transition.iloc[2, 3] = final_transition.iloc[58, 3] # Checked, completed
    intervention_transition.iloc[2, 6] = final_transition.iloc[59, 3] # Checked, completed
    intervention_transition.iloc[3, 4] = final_transition.iloc[47, 3] # Checked, completed
    intervention_transition.iloc[3, 6] = final_transition.iloc[48, 3] # Checked, completed
    intervention_transition.iloc[4, 2] = intervention_transition.iloc[0, 2] # Checked, completed
    intervention_transition.iloc[4, 5] = 0 # Checked, completed
    intervention_transition.iloc[4, 6] = intervention_transition.iloc[0, 6] # Checked, completed
    intervention_transition.iloc[4, 7] = intervention_transition.iloc[0, 7] # Checked, completed
    intervention_transition.iloc[4, 1] = 1 - intervention_transition.iloc[4, 2] - intervention_transition.iloc[4, 5] - intervention_transition.iloc[4, 6] - intervention_transition.iloc[4, 7] # Checked, completed
    
    


    # Control costs, reading sheet
    final_rewards.iloc[1, 1] = final_rewards.iloc[0, 1] + final_rewards.iloc[22, 2] # Checked, completed
    # print("Transition values (66:69, 1):")
    # print(final_transition.iloc[66:69, 1])
    # print("\nRewards values (13:16, 1):")
    # print(final_rewards.iloc[13:16, 1])
    # print("Multiplication result:")
    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[66:69, 1]).values
    reward_values = pd.to_numeric(final_rewards.iloc[13:16, 1]).values
    # print(transition_values * reward_values)
    # print("\nSum of multiplication:")
    # print((transition_values * reward_values).sum())
    final_rewards.iloc[2, 1] = (transition_values * reward_values).sum() # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[69:72, 1]).values
    reward_values = pd.to_numeric(final_rewards.iloc[16:19, 1]).values
    final_rewards.iloc[3, 1] = (transition_values * reward_values).sum() # Checked, completed
    
    df.iloc[1, 30] = final_rewards.iloc[1, 1] # Checked, completed
    df.iloc[2, 30] = final_rewards.iloc[2, 1] # Checked, completed
    df.iloc[3, 30] = final_rewards.iloc[3, 1] # Checked, completed


    # Intervention costs, reading sheet
    # Handle single value operations without .values
    final_rewards.iloc[21, 1] = pd.to_numeric(final_rewards.iloc[21, 2]) * pd.to_numeric(final_transition.iloc[63, 4]) # Checked, completed
    final_rewards.iloc[0, 2] = final_rewards.iloc[0, 1] + final_rewards.iloc[21, 1] # Checked, completed
    final_rewards.iloc[1, 2] = final_rewards.iloc[0, 1] + final_rewards.iloc[22, 2] # Checked, completed
    

    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[66:69, 2]).values
    reward_values = pd.to_numeric(final_rewards.iloc[13:16, 1]).values
    final_rewards.iloc[2, 2] = (transition_values * reward_values).sum() # Checked, completed
    

    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[69:72, 2]).values
    reward_values = pd.to_numeric(final_rewards.iloc[16:19, 1]).values
    final_rewards.iloc[3, 2] = (transition_values * reward_values).sum() # Checked, completed
    

    final_rewards.iloc[4, 2] = final_rewards.iloc[23, 2] + final_rewards.iloc[0, 1] # Checked, completed
    df.iloc[0, 31] = final_rewards.iloc[0, 2] # Checked, completed
    df.iloc[1, 31] = final_rewards.iloc[1, 2] # Checked, completed
    df.iloc[2, 31] = final_rewards.iloc[2, 2] # Checked, completed
    df.iloc[3, 31] = final_rewards.iloc[3, 2] # Checked, completed
    df.iloc[4, 31] = final_rewards.iloc[4, 2] # Checked, completed


    # Control utilities, reading sheet
    # Handle single value operations without .values
    final_rewards.iloc[25, 1] = pd.to_numeric(final_rewards.iloc[26, 2]) * pd.to_numeric(final_transition.iloc[16, 1]) + pd.to_numeric(final_rewards.iloc[25, 2]) * (1 - pd.to_numeric(final_transition.iloc[16, 1])) # Checked, completed
    final_rewards.iloc[0, 3] = final_rewards.iloc[25, 1] # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    reward_values = pd.to_numeric(final_rewards.iloc[10:13, 3]).values
    transition_values = pd.to_numeric(final_transition.iloc[63:66, 1]).values
    final_rewards.iloc[1, 3] = (reward_values * transition_values).sum() # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[66:69, 1]).values
    final_rewards.iloc[2, 3] = (reward_values * transition_values).sum() # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[69:72, 1]).values
    final_rewards.iloc[3, 3] = (reward_values * transition_values).sum() # Checked, completed
    
    final_rewards.iloc[4, 3] = final_rewards.iloc[0, 3] # Checked, completed
    df.iloc[0, 32] = final_rewards.iloc[0, 3] # Checked, completed
    df.iloc[1, 32] = final_rewards.iloc[1, 3] # Checked, completed
    df.iloc[2, 32] = final_rewards.iloc[2, 3] # Checked, completed
    df.iloc[3, 32] = final_rewards.iloc[3, 3] # Checked, completed
    df.iloc[4, 32] = final_rewards.iloc[4, 3] # Checked, completed




    # Intervention utilities, reading sheet
    final_rewards.iloc[0, 4] = final_rewards.iloc[25, 1] # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    reward_values = pd.to_numeric(final_rewards.iloc[10:13, 3]).values
    transition_values = pd.to_numeric(final_transition.iloc[63:66, 2]).values
    final_rewards.iloc[1, 4] = (reward_values * transition_values).sum() # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[66:69, 2]).values
    final_rewards.iloc[2, 4] = (reward_values * transition_values).sum() # Checked, completed
    
    # Convert to numeric and get just the values for Series operations
    transition_values = pd.to_numeric(final_transition.iloc[69:72, 2]).values
    final_rewards.iloc[3, 4] = (reward_values * transition_values).sum() # Checked, completed
    
    # Handle single value operation without .values
    final_rewards.iloc[4, 4] = pd.to_numeric(final_rewards.iloc[0, 4]) * 0.95 # Checked, completed
    df.iloc[0, 33] = final_rewards.iloc[0, 4] # Checked, completed
    df.iloc[1, 33] = final_rewards.iloc[1, 4] # Checked, completed
    df.iloc[2, 33] = final_rewards.iloc[2, 4] # Checked, completed
    df.iloc[3, 33] = final_rewards.iloc[3, 4] # Checked, completed
    df.iloc[4, 33] = final_rewards.iloc[4, 4] # Checked, completed
  
  
    # df = sheets['FinalRewards']
    # for i in range(5): 
    #     for j in range(4):
    #         print(df.iloc[i, 1 + j])
    #     print('\n')
    # print('\n')
    # df = sheets['ReadingSheet']
    # for i in range(5):
    #     for j in range(4):
    #         print(df.iloc[i, 30 + j])
    #     print('\n')
    # sheets['FinalTransition-Control'] = final_transition
    # sheets['FinalTransition-Intervention'] = intervention_transition
    # sheets['FinalRewards'] = final_rewards
    return df


def sample_beta(data_range, size):
    mean = data_range[0]
    lower = data_range[1]
    upper = data_range[2]
    
    mean_scaled = (mean - lower) / (upper - lower)
    
    var_scaled = ((upper - mean) / (2)) ** 2
    alpha = mean_scaled * ((mean_scaled * (1 - mean_scaled)) / var_scaled - 1)
    beta = (1 - mean_scaled) * ((mean_scaled * (1 - mean_scaled)) / var_scaled - 1)
    
    alpha = max(alpha, 0.1)
    beta = max(beta, 0.1)
    
    samples = np.random.beta(alpha, beta, size=size)
    
    samples = lower + samples * (upper - lower)
    return samples



def sample_uniform(data_range, size):
    _, lowerbound, upperbound = data_range
    samples = np.random.uniform(low=lowerbound, high=upperbound, size=size)
    return samples


def run_param_group(args, size, run):
    sheets = pd.read_excel(original_file, sheet_name=None)
    
    early_data = args[0]
    late_data = args[1]
    
    early_samples = sample_uniform(early_data, size)
    late_samples = sample_uniform(late_data, size)
    sum_early_late = early_samples + late_samples
    normalize = np.where(sum_early_late > 1, 1 / sum_early_late, 1)
    early_normalized = early_samples * normalize
    late_normalized = late_samples * normalize
    intermediate_normalized = 1 - early_normalized - late_normalized
    
    results = []
    for i in range(len(early_normalized)):
        early_val = early_normalized[i]
        intermediate_val = intermediate_normalized[i]
        late_val = late_normalized[i]
        sheets_copy = {k: copy.deepcopy(v) for k, v in sheets.items()}
        sheet_name = 'FinalTransition-Control' 
        if run == 'Control':
            sheets_copy[sheet_name].iloc[63, 1] = early_val
            sheets_copy[sheet_name].iloc[64, 1] = intermediate_val
            sheets_copy[sheet_name].iloc[65, 1] = late_val
        else: 
            sheets_copy[sheet_name].iloc[63, 3] = early_val
            sheets_copy[sheet_name].iloc[64, 3] = intermediate_val
            sheets_copy[sheet_name].iloc[65, 3] = late_val

        sheets_copy['ReadingSheet'] = build_reading_sheet(sheets_copy)

        # Generate random numbers once for both runs
        rng = np.random.default_rng(42)
        num_chains = 10**4
        num_steps = 100
        random_numbers = rng.random((num_chains, num_steps - 1))

        s, ageVector, ctrl_util, utilities2, ctrl_rew, rewards2 = completeRunAge(sheets_copy, input_dict, HCC, intervention=False, random_numbers=random_numbers)
        mean, median = average_death_from_hcc(s)
        avg, survival = average_hcc_and_five_year_survival(s)
        s, ageVector, int_util, utilities2, int_rew, rewards2 = completeRunAge(sheets_copy, input_dict, HCC, intervention=True, random_numbers=random_numbers)
        qaly = np.mean(int_util) - np.mean(ctrl_util)
        hcc_mask = np.any((s == 2) | (s == 3), axis=1)
        qaly_hcc = np.mean(int_util[hcc_mask]) - np.mean(ctrl_util[hcc_mask])
        if run == 'Control':
            costs = np.mean(ctrl_rew)
            costs_hcc = np.mean(ctrl_rew[hcc_mask])
        else:
            costs = np.mean(int_rew)
            costs_hcc = np.mean(int_rew[hcc_mask])
        icer = (np.mean(int_rew) - np.mean(ctrl_rew)) / qaly
        nmb = 100000 * (np.mean(int_util) - np.mean(ctrl_util)) - (np.mean(int_rew) - np.mean(ctrl_rew))
        results.append([early_val, late_val, intermediate_val, icer, nmb, mean, median, survival, qaly, qaly_hcc, costs, costs_hcc])

    df = pd.DataFrame(results, columns=['early_val', 'late_val', 'intermediate_val', 'icer', 'nmb', 'mean death', 'median death', '5 year survival', 'raw qaly diff per person', 'raw qaly diff per hcc person'
    'raw cost per person', 'raw cost per hcc person'])
    
    output_file = os.path.join(output_base, f'{run.lower()}.csv')
    print(f"\nSaving results for {run} to: {output_file}")
    print(f"Results shape: {df.shape}")
    
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved file to {output_file}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise
    
    return run


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    control_data = [[.15, .05, .25], [.75, .55, .95]]
    intervention_data = [[.81, .6075, 1], [.11, .0825, .1375]]
    size = 10000
    jobs = []
    jobs.append((control_data, size, 'Control'))
    jobs.append((intervention_data, size, 'Intervention'))
    # df = pd.read_excel(original_file, sheet_name='ReadingSheet')
    # for i in range(6):
    #     for j in range(4):
    #         print(df.iloc[i, 30 + j])
        
    # df = pd.read_excel(original_file, sheet_name = 'FinalRewards')
    # for i in range(6): 
    #     for j in range(4):
    #         print(df.iloc[i, 1 + j])
    print(f"Running sensitivity analysis on {len(jobs)} variables with {size} samples each")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_param_group, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running variables"):
            try:
                future.result()
            except Exception as e:
                import traceback
                print(f"\n❌ Error in job {future}:")
                traceback.print_exception(type(e), e, e.__traceback__)


    print("Generating plots")
    two_way_plot()
