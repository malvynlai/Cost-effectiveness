import numpy as np
import pandas as pd
from numba import njit, prange
from datetime import datetime

# Set display options for NumPy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Set display options for pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from scipy import stats
import compressedInputsRead as readInputs
import MCMC_improved_cleanv3_ver as MC
import sys
import time
import Visualizations.vizFinal as viz
import Visualizations.verFinal as viz2

np.random.seed(42)

# file = r"C:\Users\sovan\Box\Sovann Linden's Files\Cost-effectiveness\Inputs\Inputs_CEA_v4.xlsx"
file = '/Users/malvynlai/Library/CloudStorage/Box-Box/Cost-effectiveness/Inputs/Inputs_CEA_v4_sensitivity.xlsx'
file = '/Users/malvynlai/Desktop/Lab Projects/Cost Effectiveness/Inputs/Inputs_CEA_v4_changed HCC distributions_JLcopy_3.19.25_single input.xlsx'
file = '/Users/malvynlai/Library/CloudStorage/Box-Box/Cost-effectiveness/Inputs/Inputs_CEA_v4_5.8.25.xlsx'
# file = '/home/ec2-user/Cost-effectiveness/Inputs/Inputs_CEA_v4.xlsx'
#file = '~/Desktop/Lab Projects/Cost Effectiveness/Inputs/Inputs_CEA_v4.xlsx'
# file = r"C:\Users\joannekl\Box\Cost-effectiveness\Inputs\Inputs_CEA_v4_2.20.xlsx" 

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


def completeRun(file, input_dict, HCC, intervention=False):
    df = pd.read_excel(file, sheet_name='ReadingSheet')  
    discountRate = 0.03
    num_steps = 100
    num_chains = 10**5

    """t = readInputs.generateFinalTransition(df, input_dict, HCC).to_numpy()"""
    if intervention:
        df = pd.read_excel(file, sheet_name='FinalTransition-Intervention')
    t = readInputs.generateCleanTransition(df).to_numpy()
    if intervention:
        df = pd.read_excel(file, sheet_name='ReadingSheet')
        data = {
        'Intervention Cost': [df.iloc[0, 31], df.iloc[1, 31], df.iloc[2, 31], df.iloc[3, 31], df.iloc[4, 31], df.iloc[5, 31], df.iloc[6, 31]],
        'Intervention Utility': [df.iloc[0, 33], df.iloc[1, 33], df.iloc[2, 33], df.iloc[3, 33], df.iloc[4, 33], df.iloc[5, 33], df.iloc[6, 33]],
        }
        index = ['MASLD', 'HCC', 'Treated', 'Untreated', 'False Positive HCC', 'Death', 'Cirrhosis']
        costUtility = pd.DataFrame(data, index=index)
    else: 
        costUtility = readInputs.generate_cost_utility_matrix(df)
    rewardVector = np.expand_dims(costUtility.iloc[:,0].to_numpy(), axis = 1)
    utilityVector = np.expand_dims(costUtility.iloc[:,1].to_numpy(), axis = 1)

    checkTypes(t,rewardVector,utilityVector)
    triple =  (discountRate, num_steps, num_chains)
    states, utilities, rewards = MC.realLifeSimulation(t, rewardVector, utilityVector, *triple)
    return states, [], utilities, rewards


def completeRunAge(sheets, input_dict, HCC, intervention=False, random_numbers=None):
    np.random.seed(42)
    df = sheets['ReadingSheet']
    discountRate = 0.03
    num_steps = 100
    num_chains = 10**5

    
    random_numbers = np.random.rand(num_chains, num_steps - 1)
    
    posDeath = 5
    ageRange = 83

    df1 = sheets['ActuarialTables']  
    probDeath = readInputs.generate_prob_death(df1)
    df2 = sheets['AgeVector']
    ageVector = readInputs.generate_age_vector(df2,num_chains)
    
    if intervention:
        df = sheets['FinalTransition-Intervention']
    t = readInputs.generateCleanTransition(df).to_numpy()

    sizeAges = len(t)
    
    t = MC.generate_transition_matrixAgeDistribution(t, sizeAges, probDeath, posDeath)
    
    
    if intervention:
        df = sheets['ReadingSheet']
        data = {
        'Intervention Cost': [df.iloc[0, 31], df.iloc[1, 31], df.iloc[2, 31], df.iloc[3, 31], df.iloc[4, 31], df.iloc[5, 31], df.iloc[6, 31]],
        'Intervention Utility': [df.iloc[0, 33], df.iloc[1, 33], df.iloc[2, 33], df.iloc[3, 33], df.iloc[4, 33], df.iloc[5, 33], df.iloc[6, 33]],
        }
        index = ['MASLD', 'HCC', 'Treated', 'Unreated', 'False Positive HCC', 'Death', 'Cirrhosis']
        costUtility = pd.DataFrame(data, index=index)
    else: 
        costUtility = readInputs.generate_cost_utility_matrix(df)

    rewardVector = np.expand_dims(costUtility.iloc[:,0].to_numpy(), axis = 1)
    utilityVector = np.expand_dims(costUtility.iloc[:,1].to_numpy(), axis = 1)
    # print("Start checking")

    # checkTypesAges(t,rewardVector,utilityVector,ageRange)

    states, utilities, utilities2, rewards, rewards2 = MC.simulationwithAge(
        t, 
        rewardVector, 
        utilityVector, 
        ageVector, 
        random_numbers, 
        discountRate, 
        num_steps, 
        num_chains
    )

    # # Print transition probabilities for different ages
    # for age in [0, 20, 40, 60]:  # Sample ages
    #     age_block = t[0, age*sizeAges:(age+1)*sizeAges]
    #     print(f"\nAge {age+18} transition probabilities from MASLD:")
    #     print(f"To MASLD: {age_block[0]:.4f}")
    #     print(f"To HCC: {age_block[1]:.4f}")
    #     print(f"To Death: {age_block[5]:.4f}")
    return states, ageVector, utilities, utilities2, rewards, rewards2


@njit
def hcc_total(states):
    total = 0
    num_patients = states.shape[0]
    for i in range(num_patients):
        for j in range(states.shape[1]):
            if states[i,j] == 2 or states[i, j] == 3:
                total += 1
                break
    return total / num_patients


@njit
def cirrhosis_total(states):
    total = 0
    num_patients = states.shape[0]
    for i in range(num_patients):
        for j in range(states.shape[1]):
            if states[i,j] == 6:
                total += 1
                break
    return total / num_patients


@njit(parallel=True)
def average_cirrhosis(states, ageVector):
    cirrhosis_ages = np.zeros(states.shape[0])
    total = 0
    for i in prange(states.shape[0]):
        row = states[i]  
        if 6 in row:
            first_cirrhosis = np.where(row == 6)[0][0]
            cirrhosis_ages[i] = first_cirrhosis + ageVector[i]
            total += 1
    return np.sum(cirrhosis_ages) / total + 18 if len(cirrhosis_ages) > 0 else 0


@njit(parallel=True)
def average_hcc_and_five_year_survival(states, ageVector):
    n_patients = states.shape[0]
    has_hcc = np.zeros(n_patients, dtype=np.int32)
    hcc_age = np.zeros(n_patients, dtype=np.int32)
    survived = np.zeros(n_patients, dtype=np.int32)
    
    for i in prange(n_patients):  
        for j in range(states.shape[1]):
            if states[i,j] == 2 or states[i,j] == 3:
                has_hcc[i] = 1
                hcc_age[i] = j + ageVector[i]
                survived[i] = 1  
                for k in range(5):
                    if j + k < states.shape[1] and states[i, j + k] == 5:
                        survived[i] = 0  
                        break
                break
    hcc_ages = hcc_age[has_hcc == 1]
    survival_rate = np.sum(survived[has_hcc == 1]) / np.sum(has_hcc) if np.sum(has_hcc) > 0 else 0
    
    return np.mean(hcc_ages) + 18 if len(hcc_ages) > 0 else 0, survival_rate


@njit(parallel=True)
def average_death_from_hcc(states):
    n_patients = states.shape[0]
    death_times = np.zeros(n_patients, dtype=np.int32)
    has_hcc_death = np.zeros(n_patients, dtype=np.int32)
    
    for i in prange(n_patients):
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


@njit(parallel=True)
def average_death_from_masld(states, ageVector):
    n_patients = states.shape[0]
    death_ages = np.zeros(n_patients, dtype=np.int32)
    is_masld_death = np.zeros(n_patients, dtype=np.int32)
    
    for i in prange(n_patients):
        row = states[i]  
        if 1 not in row and 5 in row:
            death_idx = np.where(row == 5)[0][0]
            if death_idx > 0 and row[death_idx - 1] == 0:
                death_ages[i] = death_idx + ageVector[i]
                is_masld_death[i] = 1
    valid_ages = death_ages[is_masld_death == 1]
    return np.mean(valid_ages) + 18 if len(valid_ages) > 0 else 0


@njit(parallel=True)
def ten_year_cirrhosis(states):
    total_patients = states.shape[0]
    cirrhosis_count = 0
    for i in prange(total_patients):
        for j in range(11):
            if states[i, j] == 6:
                cirrhosis_count += 1
                break
    return cirrhosis_count / total_patients


@njit(parallel=True)
def ten_year_hcc(states):
    hcc_count = 0
    total_patients = states.shape[0]
    for i in prange(total_patients):
        for j in range(11):
            if states[i, j] == 2 or states[i,j] == 3:
                hcc_count += 1
                break
    return hcc_count / total_patients





input_dict = {
        'cirrhosisUnderdiagnosisRateInMasld_Rate': 0.059,
        'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD': 0.043,
        'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic': 0.0011,
        'masldIncidenceRates_falsePositiveHCC': 0.1,
        'masldIncidenceRates_masldToCirrhosis': 0.006}
HCC = np.array([0.457,0.23,0.313])


def main():
    # s = completeRun(file, input_dict, HCC)
    num_runs = 1

    columns = [f'Run {i+1}' for i in range(num_runs)] + ['Averages']

    results_no_intervention = pd.DataFrame(np.nan, 
        index = ['Developed HCC %', 'Developed Cirrhosis %', 'Avg. Age HCC (At HCC Development)', 'Avg. Age Cirrhosis', 'Mean Death from HCC',
        'Median Death From HCC', 'Avg. HCC Rewards (Only HCC Nodes, Treated + Untreated)', 'Avg. Patient Rewards (Over Entire Lifetime)', 'Avg Death Age MASLD', '10 Year Cirrhosis', 
        '10 Year HCC', '5 Year HCC Survival %', 'Avg. HCC Utilities', 'Avg. Patient Utilities', 'ICER', 'Avg. MASLD Rewards (MASLD Nodes)', 'Avg. False Positive Rewards (Only False Positive Node)', 
        'Avg. Treated HCC Rewards (Only Treated Node)', 'Avg. Untreated HCC Rewards (Only Treated Node)'],
        columns = columns
    )

    results_intervention = pd.DataFrame(np.nan,
        index = ['Developed HCC %', 'Developed Cirrhosis %', 'Avg. Age HCC (At HCC Development)', 'Avg. Age Cirrhosis', 'Mean Death from HCC',
        'Median Death From HCC', 'Avg. HCC Rewards (Only HCC Nodes, Treated + Untreated)', 'Avg. Patient Rewards (Over Entire Lifetime)', 'Avg Death Age MASLD', '10 Year Cirrhosis', 
        '10 Year HCC', '5 Year HCC Survival %', 'Avg. HCC Utilities', 'Avg. Patient Utilities', 'ICER', 'Avg. MASLD Rewards (MASLD Nodes)', 'Avg. False Positive Rewards (Only False Positive Node)', 
        'Avg. Treated HCC Rewards (Only Treated Node)', 'Avg. Untreated HCC Rewards (Only Treated Node)'],
        columns = columns
    )


    sheets = pd.read_excel(file, sheet_name=None)
    for i in range(num_runs):
        s, ageVector, utilities, utilities2, rewards, rewards2 = completeRunAge(sheets, input_dict, HCC, intervention=False)
        s = pd.DataFrame(s)
        s.to_csv('states.csv')
        mean, median = average_death_from_hcc(s.to_numpy())
        masld_deaths = average_death_from_masld(s.to_numpy(), ageVector)
        average_hcc, survival = average_hcc_and_five_year_survival(s.to_numpy(), ageVector)
        average_cirrhosis_result = average_cirrhosis(s.to_numpy(), ageVector)
        s = np.array(s) 
        
        hcc_mask = np.where((s == 2) | (s == 3))
        hcc_count = np.sum(np.any((s == 2) | (s == 3), axis=1))
        hcc_mean_rewards = np.sum(rewards2[hcc_mask]) / hcc_count
        hcc_mean_utilities = np.sum(utilities2[hcc_mask]) / hcc_count

        
        masld_mask = np.where((s == 0) | (s == 4))
        masld_total = np.sum(np.any(s == 0, axis=1))
        masld_mean_rewards = np.sum(rewards2[masld_mask]) / masld_total
        
        untreated_mask = np.where(s == 3)  
        untreated_total = np.sum(np.any(s == 3, axis=1))
        untreated_mean_rewards = np.sum(rewards2[untreated_mask]) / untreated_total
        
        treated_mask = np.where(s == 2)  
        treated_total = np.sum(np.any(s == 2, axis=1))
        treated_mean_rewards = np.sum(rewards2[treated_mask]) / treated_total

        false_positive_mask = np.where((s == 4))
        false_positive_total = np.sum(np.any(s == 4, axis=1)) 
        false_positive_mean_rewards = np.sum(rewards2[false_positive_mask]) / false_positive_total

        ten_year_cirrhosis_rate = ten_year_cirrhosis(s)
        ten_year_hcc_rate = ten_year_hcc(s)

 
        results_no_intervention.iloc[0, i] = hcc_total(s)
        results_no_intervention.iloc[1, i] = cirrhosis_total(s)
        results_no_intervention.iloc[2, i] = average_hcc
        results_no_intervention.iloc[3, i] = average_cirrhosis_result
        results_no_intervention.iloc[4, i] = mean
        results_no_intervention.iloc[5, i] = median
        results_no_intervention.iloc[6, i] = hcc_mean_rewards
        results_no_intervention.iloc[7, i] = np.mean(rewards)
        results_no_intervention.iloc[8, i] = masld_deaths
        results_no_intervention.iloc[9, i] = ten_year_cirrhosis_rate
        results_no_intervention.iloc[10, i] = ten_year_hcc_rate
        results_no_intervention.iloc[11, i] = survival
        results_no_intervention.iloc[12, i] = hcc_mean_utilities
        results_no_intervention.iloc[13, i] = np.mean(utilities)
        results_no_intervention.iloc[14, i] = (results_intervention.iloc[7, i] - results_no_intervention.iloc[7, i])/(results_intervention.iloc[13, i] - results_no_intervention.iloc[13, i])
        results_no_intervention.iloc[15, i] = masld_mean_rewards
        results_no_intervention.iloc[16, i] = 0
        results_no_intervention.iloc[17, i] = untreated_mean_rewards
        results_no_intervention.iloc[18, i] = treated_mean_rewards

        s, ageVector, utilities, utilities2, rewards, rewards2 = completeRunAge(sheets, input_dict, HCC, intervention=True)
        s = pd.DataFrame(s)

        mean, median = average_death_from_hcc(s.to_numpy())
        masld_deaths = average_death_from_masld(s.to_numpy(), ageVector)
        average_hcc, survival = average_hcc_and_five_year_survival(s.to_numpy(), ageVector)
        average_cirrhosis_result = average_cirrhosis(s.to_numpy(), ageVector)

        s = np.array(s)  
        
        hcc_mask = np.where((s == 2) | (s == 3))
        hcc_count = np.sum(np.any((s == 2) | (s == 3), axis=1))
        hcc_mean_rewards = np.sum(rewards2[hcc_mask]) / hcc_count
        hcc_mean_utilities = np.sum(utilities2[hcc_mask]) / hcc_count

        ten_year_cirrhosis_rate = ten_year_cirrhosis(s)
        ten_year_hcc_rate = ten_year_hcc(s)

        masld_mask = np.where((s == 0) | (s == 4))
        masld_total = np.sum(np.any(s == 0, axis=1))
        masld_mean_rewards = np.sum(rewards2[masld_mask]) / masld_total
        
        treated_mask = np.where(s == 3)  
        treated_total = np.sum(np.any(s == 3, axis=1))
        treated_mean_rewards = np.sum(rewards2[treated_mask]) / treated_total
        
        untreated_mask = np.where(s == 2)  
        untreated_total = np.sum(np.any(s == 2, axis=1))
        untreated_mean_rewards = np.sum(rewards2[untreated_mask]) / untreated_total

        false_positive_mask = np.where((s == 4))
        false_positive_mean_rewards = np.sum(rewards2[false_positive_mask]) / 100000

        results_intervention.iloc[0, i] = hcc_total(s)
        results_intervention.iloc[1, i] = cirrhosis_total(s)
        results_intervention.iloc[2, i] = average_hcc
        results_intervention.iloc[3, i] = average_cirrhosis_result
        results_intervention.iloc[4, i] = mean
        results_intervention.iloc[5, i] = median
        results_intervention.iloc[6, i] = hcc_mean_rewards
        results_intervention.iloc[7, i] = np.mean(rewards)
        results_intervention.iloc[8, i] = masld_deaths
        results_intervention.iloc[9, i] = ten_year_cirrhosis_rate
        results_intervention.iloc[10, i] = ten_year_hcc_rate
        results_intervention.iloc[11, i] = survival
        results_intervention.iloc[12, i] = hcc_mean_utilities
        results_intervention.iloc[13, i] = np.mean(utilities)
        results_intervention.iloc[14, i] = (results_intervention.iloc[7, i] - results_no_intervention.iloc[7, i])/(results_intervention.iloc[13, i] - results_no_intervention.iloc[13, i])
        results_intervention.iloc[15, i] = masld_mean_rewards
        results_intervention.iloc[16, i] = false_positive_mean_rewards
        results_intervention.iloc[17, i] = treated_mean_rewards
        results_intervention.iloc[18, i] = untreated_mean_rewards

    results_no_intervention['Averages'] = results_no_intervention.iloc[:, :50].mean(axis=1)
    results_intervention['Averages'] = results_intervention.iloc[:, :10].mean(axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    control_sheet = f'Control_{timestamp}'
    intervention_sheet = f'Intervention_{timestamp}'

    try:
        with pd.ExcelWriter('verifications.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            results_no_intervention.to_excel(writer, sheet_name='Control', index=True)
            results_intervention.to_excel(writer, sheet_name='Intervention', index=True)
    except FileNotFoundError:
        with pd.ExcelWriter('verifications.xlsx', engine='openpyxl') as writer:
            results_no_intervention.to_excel(writer, sheet_name='Control', index=True)
            results_intervention.to_excel(writer, sheet_name='Intervention', index=True)

    # average_hcc, survival = average_hcc_and_five_year_survival(s, ageVector)
    # print(f'Average Age of Developing HCC: {average_hcc}')
    # print(f'Average Age of Developing Cirrhosis: {average_cirrhosis(s, ageVector)}')
    # mean, median = average_death_from_hcc(s)
    # print(f'Mean Years Before Death After HCC: {mean}')
    # print(f'Median Death From HCC: {median}')
    # # print(utilities.shape)
    # hcc_mask = s.apply(lambda row: 1 in row.values, axis=1).values
    # hcc_mean_rewards = np.mean(rewards[hcc_mask])
    # print(f"Mean Rewards For Patients Who Develop HCC: {hcc_mean_rewards}")
    # print(f"Mean Rewards For All Patients: {np.mean(rewards)}")
    # print(f'Average Death From MASLD {average_death_from_masld(s, ageVector)}')
    # print(f'Ten Year Cirrhosis Incidence Rate: {ten_year_cirrhosis(s)}')
    # print(f'Ten Year HCC Incidence Rate: {ten_year_hcc(s)}')
    # print(f'Five Year HCC Survival: {survival}')

    intervention = pd.DataFrame(s)
    # print(results_intervention)
    # viz.graphs(control)
    # viz2.graphs(control, intervention)


if __name__ == '__main__':
    main()