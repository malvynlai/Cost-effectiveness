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
output_base = "/sailhome/malvlai/Cost-effectiveness/Results/HCC Distributions/Data"

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


def run_param_group(df, args, size, run):
    sheets = pd.read_excel(original_file, sheet_name=None)
    early = args[0]
    late = args[1]
    
    if (early + late) >  1: 
        normalize = 1 / (early + late)
        early *= normalize
        late *=normalize
    
    intermediate = 1 - early - late
    
    
    sheets_copy = {k: copy.deepcopy(v) for k, v in sheets.items()}
    sheet_name = 'FinalTransition-Control' 
    if run == 'Control':
        sheets_copy[sheet_name].iloc[63, 1] = early
        sheets_copy[sheet_name].iloc[64, 1] = intermediate
        sheets_copy[sheet_name].iloc[65, 1] = late
    else: 
        sheets_copy[sheet_name].iloc[63, 3] = early
        sheets_copy[sheet_name].iloc[64, 3] = intermediate
        sheets_copy[sheet_name].iloc[65, 3] = late

    sheets_copy['ReadingSheet'] = build_reading_sheet(sheets_copy)

    # Generate random numbers once for both runs
    rng = np.random.default_rng(42)
    num_chains = 10**5
    num_steps = 100
    random_numbers = rng.random((num_chains, num_steps - 1))

    s, ageVector, ctrl_util, _, ctrl_rew, _ = completeRunAge(sheets_copy, input_dict, HCC, intervention=False, random_numbers=random_numbers)
    mean, median = average_death_from_hcc(s)
    avg, survival = average_hcc_and_five_year_survival(s)
    s, ageVector, int_util, _, int_rew, _ = completeRunAge(sheets_copy, input_dict, HCC, intervention=True, random_numbers=random_numbers)

    icer = (np.mean(int_rew) - np.mean(ctrl_rew)) / (np.mean(int_util) - np.mean(ctrl_util))
    nmb = 100000 * (np.mean(int_util) - np.mean(ctrl_util)) - (np.mean(int_rew) - np.mean(ctrl_rew))
    results = [early, late, intermediate, icer, mean, median, survival]

    return results


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    control_data = [[.05, .1, .15, .2, .25], [.55, .65, .75, .85, .95]]
    intervention_data = []
    df = pd.DataFrame(columns=['early_val', 'late_val', 'intermediate_val', 'icer', 'mean death', 'median death', 'survival rate']) 
    for i in range(5):
        for j in range(5):
            data = [control_data[0][i], control_data[1][j]]
            results = run_param_group(df, data, 1, 'Control')
            df.loc[len(df)] = results
    

    output_file = os.path.join(output_base, 'control.csv')
    df.to_csv(output_file)

    # Graphing section

