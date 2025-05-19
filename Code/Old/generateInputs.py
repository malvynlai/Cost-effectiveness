# TO DO: Run all of transition matrix, utility vector, and reward vector combinations through realLifeSimulation

import pandas as pd 
import numpy as np 
import itertools
import json 
import random
import time

def read_spreadsheet_readingsheet(file_path):
    return pd.read_excel(file_path, sheet_name='ReadingSheet')    

def read_spreadsheet_finalrewards(file_path):
    return pd.read_excel(file_path, sheet_name='FinalRewards')  


def generateFinalTransition(df, input_dict, HCC):
    masld_hcc = df.iloc[1,1]
    masld_treatment = df.iloc[2,1]
    masld_treated = df.iloc[3,1]
    masld_falsePositiveHCC = df.iloc[4,1]
    masld_death = df.iloc[5,1]
    masld_cirrhosis = df.iloc[6,1]

    hcc_falsePositiveHCC = df.iloc[4,2]
    hcc_death = df.iloc[5,2]
    hcc_cirrhosis = df.iloc[6,2]

    treatment_treatment = df.iloc[2,3]
    treatment_treated = df.iloc[3,3]
    treatment_falsePositiveHCC = df.iloc[4,3]
    treatment_death = df.iloc[5,3]
    treatment_cirrhosis = df.iloc[6,3]

    cirrhosis_hcc = df.iloc[1,7]
    cirrhosis_treatment = df.iloc[2,7]
    cirrhosis_treated = df.iloc[3,7]
    cirrhosis_falsePositiveHCC = df.iloc[4,7]
    cirrhosis_death = df.iloc[5,7]
    cirrhosis_cirrhosis = df.iloc[6,7]


    hccOutcomesRatestreated = np.array(df.iloc[0, 17], df.iloc[3, 17],df.iloc[6, 17])
    treatment_hcc = np.dot(hccOutcomesRatestreated, HCC)
    hccOutcomesRatesDeath = np.array(df.iloc[1,17],df.iloc[4, 17],df.iloc[7,17])
    death_hcc =np.dot(hccOutcomesRatesDeath, HCC)

    data = {
        "Control Matrix": ["MASLD", "HCC", "Treatment", "Treated", "False Positive HCC", "Death", "Cirrhosis"],
        "MASLD": [masld_masld, masld_hcc, masld_treatment, masld_treated, masld_falsePositiveHCC, masld_death, masld_cirrhosis],
        "HCC": [hcc_masld, hcc_hcc, hcc_treatment, hcc_treated, hcc_falsePositiveHCC, hcc_death, hcc_cirrhosis],
        "Treatment": [treatment_masld, treatment_hcc, treatment_treatment, treatment_treated, treatment_falsePositiveHCC, treatment_death, treatment_cirrhosis],
        "Treated": [treated_masld, treated_hcc, treated_treatment, treated_treated, treated_falsePositiveHCC, treated_death, treated_cirrhosis],
        "False Positive HCC": [falsePositiveHCC_masld, falsePositiveHCC_hcc, falsePositiveHCC_treatment, falsePositiveHCC_treated, falsePositiveHCC_falsePositiveHCC, falsePositiveHCC_death, falsePositiveHCC_cirrhosis],
        "Death": [death_masld, death_hcc, death_treatment, death_treated, death_falsePositiveHCC, death_death, death_cirrhosis],
        "Cirrhosis": [cirrhosis_masld, cirrhosis_hcc, cirrhosis_treatment, cirrhosis_treated, cirrhosis_falsePositiveHCC, cirrhosis_death, cirrhosis_cirrhosis]
    }
    transition_matrix = pd.DataFrame(data)
    return transition_matrix
    



def generate_final_transition(df, input_dict):
    hccDistributionEarly = input_dict['hccDistributionEarly']
    hccDistributionIntermediate = input_dict['hccDistributionIntermediate']
    hccDistributionLate = input_dict['hccDistributionLate']
    cirrhosisUnderdiagnosisRateInMasld_Rate = input_dict['cirrhosisUnderdiagnosisRateInMasld_Rate']
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD = input_dict['cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD']
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic = input_dict['cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic']
    masldIncidenceRates_falsePositiveHCC = input_dict['masldIncidenceRates_falsePositiveHCC']
    masldIncidenceRates_masldToCirrhosis = input_dict['masldIncidenceRates_masldToCirrhosis']

    # the first word in the variable name represents the column, second represents row
    masldIncidenceRates_masldToHCC = cirrhosisUnderdiagnosisRateInMasld_Rate * cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD + (1 - cirrhosisUnderdiagnosisRateInMasld_Rate) * cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic
    hcc_masld = masldIncidenceRates_masldToHCC

    cirrhosisUnderdiagnosisRateInMasld_MasldToDeathForUD = df.iloc[3,13]
    cirrhosisUnderdiagnosisRateInMasld_MasldToDeathForNoncirrhotic = df.iloc[4,13]
    masldIncidenceRates_masldToDeath = cirrhosisUnderdiagnosisRateInMasld_Rate  + (1 - cirrhosisUnderdiagnosisRateInMasld_Rate) * cirrhosisUnderdiagnosisRateInMasld_MasldToDeathForNoncirrhotic
    death_masld = masldIncidenceRates_masldToDeath

    falsePositiveHCC_masld = masldIncidenceRates_falsePositiveHCC

    cirrhosis_masld = masldIncidenceRates_masldToCirrhosis

    masld_masld = 1 - hcc_masld - death_masld - falsePositiveHCC_masld - cirrhosis_masld
    masld_hcc = df.iloc[1,1]
    masld_treatment = df.iloc[2,1]
    masld_treated = df.iloc[3,1]
    masld_falsePositiveHCC = df.iloc[4,1]
    masld_death = df.iloc[5,1]
    masld_cirrhosis = df.iloc[6,1]
# HCC column 
    
    hccOutcomesRates_early_treated = df.iloc[0, 17]
    hccOutcomesRates_intermediate_treated = df.iloc[3, 17]
    hccOutcomesRates_late_treated = df.iloc[6, 17]
    hccOutcomesRates_weighted_treated = hccOutcomesRates_early_treated * hccDistributionEarly + hccOutcomesRates_intermediate_treated * hccDistributionIntermediate + hccOutcomesRates_late_treated * hccDistributionLate
    treatment_hcc = hccOutcomesRates_weighted_treated #done

    hccOutcomesRates_early_death = df.iloc[1,17]
    hccOutcomesRates_intermediate_death = df.iloc[4, 17]
    hccOutcomesRates_late_death = df.iloc[7,17]
    hccOutcomesRates_weighted_death = hccOutcomesRates_early_death * hccDistributionEarly + hccOutcomesRates_intermediate_death * hccDistributionIntermediate + hccOutcomesRates_late_death * hccDistributionLate
    death_hcc = hccOutcomesRates_weighted_death #done

    hcc_hcc = 1 - treatment_hcc - death_hcc 
    hcc_treatment = df.iloc[2,2]
    
    treatedOutcomesRates_treatedAfterEarly_death = df.iloc[1,22]
    treatedOutcomesRates_treatedAfterEarly_treated = df.iloc[0,22]
    treatedOutcomesRates_treatedAfterEarly_reccurence = 1 - treatedOutcomesRates_treatedAfterEarly_death - treatedOutcomesRates_treatedAfterEarly_treated
    
    treatedOutcomesRates_treatedAfterIntermediate_death = df.iloc[4, 22]
    treatedOutcomesRates_treatedAfterIntermediate_treated = df.iloc[3,22]
    treatedOutcomesRates_treatedAfterIntermediate_reccurence = 1 - treatedOutcomesRates_treatedAfterIntermediate_death - treatedOutcomesRates_treatedAfterIntermediate_treated

    treatedOutcomesRates_treatedAfterLate_death = df.iloc[7, 22]
    treatedOutcomesRates_treatedAfterLate_treated = df.iloc[6, 22]
    treatedOutcomesRates_treatedAfterLate_reccurence = 1 - treatedOutcomesRates_treatedAfterLate_death - treatedOutcomesRates_treatedAfterLate_treated

    treatedOutcomesRates_weighted_recurrence = treatedOutcomesRates_treatedAfterEarly_reccurence * hccDistributionEarly + treatedOutcomesRates_treatedAfterIntermediate_reccurence * hccDistributionIntermediate + treatedOutcomesRates_treatedAfterLate_reccurence * hccDistributionLate
    hcc_treated = treatedOutcomesRates_weighted_recurrence
    
    
    hcc_falsePositiveHCC = df.iloc[4,2] 
    hcc_death = df.iloc[5,2]
    hcc_cirrhosis = df.iloc[6,2] #done

# Treatment column
    treatment_masld = df.iloc[0,3]
    
    
    treatment_treatment = df.iloc[2,3]
    treatment_treated = df.iloc[3,3]
    treatment_falsePositiveHCC = df.iloc[4,3]
    treatment_death = df.iloc[5,3]
    treatment_cirrhosis = df.iloc[6,3]

# Treated column
    treated_masld = df.iloc[0,4]
    treated_hcc = df.iloc[1,4]
    treated_treatment = df.iloc[2,4]

    treatedOutcomesRates_weighted_treated = treatedOutcomesRates_treatedAfterEarly_treated * hccDistributionEarly + treatedOutcomesRates_treatedAfterIntermediate_treated * hccDistributionIntermediate + treatedOutcomesRates_treatedAfterLate_treated * hccDistributionLate
    treated_treated = treatedOutcomesRates_weighted_treated
    treated_falsePositiveHCC = df.iloc[4,4]
    treated_death = df.iloc[5,4]
    treated_cirrhosis = df.iloc[6,4]

# FalsePositiveHCC column
    falsePositiveHCC_hcc = df.iloc[1,5]
    falsePositiveHCC_treatment = df.iloc[2,5]
    falsePositiveHCC_treated = df.iloc[3,5]
    falsePositiveHCC_falsePositiveHCC = df.iloc[4,5]
    falsePositiveHCC_death = df.iloc[5,5]
    falsePositiveHCC_cirrhosis = df.iloc[6,5]

# Death column
    
    death_treatment = df.iloc[2,6]
    treatedOutcomesRates_weighted_death = treatedOutcomesRates_treatedAfterEarly_death * hccDistributionEarly + treatedOutcomesRates_treatedAfterIntermediate_death * hccDistributionIntermediate + treatedOutcomesRates_treatedAfterLate_death * hccDistributionLate
    death_treated = treatedOutcomesRates_weighted_death
    death_falsePositiveHCC = df.iloc[4,6]
    death_death = df.iloc[5,6]
    death_cirrhosis = df.iloc[6,6]

# Cirrhosis column
    cirrhosis_hcc = df.iloc[1,7]
    cirrhosis_treatment = df.iloc[2,7]
    cirrhosis_treated = df.iloc[3,7]
    cirrhosis_falsePositiveHCC = df.iloc[4,7]
    cirrhosis_death = df.iloc[5,7]
    cirrhosis_cirrhosis = df.iloc[6,7]

    data = {
        "Control Matrix": ["MASLD", "HCC", "Treatment", "Treated", "False Positive HCC", "Death", "Cirrhosis"],
        "MASLD": [masld_masld, masld_hcc, 0, 0, masld_falsePositiveHCC, masld_death, masld_cirrhosis],
        "HCC": [0, hcc_hcc, hcc_treatment, 0, 0, hcc_death, 0],
        "Treatment": [0, 0, 0, treatment_treated, 0, treatment_death, 0],
        "Treated": [0,0,0, treated_treated, 0, treated_death, 0],
        "False Positive HCC": [1,0,0,0,0,0,0],
        "Death": [0,0,0,0,0,1,0],
        "Cirrhosis": [0,0,0,0,0, cirrhosis_cirrhosis]
    }
    transition_matrix = pd.DataFrame(data)

    return transition_matrix

def generate_cost_utility(df, transition_matrix, hccStages_costUtility, inputs):
    # Control Cost Column
    controlCost_masld = 3537
    controlCost_hcc = inputs['hccDistributionEarly'] * hccStages_costUtility['Control Cost'][0] + inputs['hccDistributionIntermediate'] * hccStages_costUtility['Control Cost'][1] + inputs['hccDistributionLate'] * hccStages_costUtility['Control Cost'][2]
    controlCost_treatment = controlCost_hcc 
    controlCost_treated = 0 
    controlCost_falsePositiveHCC = controlCost_masld + 554 
    controlCost_death = 0 
    controlCost_cirrhosis = 0 
    controlCost_column = [controlCost_masld, controlCost_hcc, controlCost_treatment, controlCost_treated, controlCost_falsePositiveHCC, controlCost_death, controlCost_cirrhosis]
    
    # Intervention Cost column
    # I am hard coding the HCC distribution for "Screen" but can change this if Sovann wants
    screen_hccDistribution_early = 0.707
    screen_hccDistribution_intermediate = 0.156
    screen_hccDistribution_late = 0.137
    
    interventionCost_masld = controlCost_masld 
    interventionCost_hcc = screen_hccDistribution_early * hccStages_costUtility['Intervention Cost'][0] + screen_hccDistribution_intermediate * hccStages_costUtility['Intervention Cost'][1] + screen_hccDistribution_late * hccStages_costUtility['Intervention Cost'][2] 
    interventionCost_treatment = interventionCost_hcc
    interventionCost_treated = 0 
    interventionCost_falsePositiveHCC = controlCost_falsePositiveHCC
    interventionCost_death = 0
    interventionCost_cirrhosis = 0 
    interventionCost_column = [interventionCost_masld, interventionCost_hcc, interventionCost_treatment, interventionCost_treated, interventionCost_falsePositiveHCC, interventionCost_death, interventionCost_cirrhosis]

    # Control Utility column 
    controlUtility_masld = 0.85 
    controlUtility_hcc = inputs['hccDistributionEarly'] * hccStages_costUtility['Control Utility'][0] + inputs['hccDistributionIntermediate'] * hccStages_costUtility['Control Utility'][1] + inputs['hccDistributionLate'] * hccStages_costUtility['Control Utility'][2]
    controlUtility_treatment = controlUtility_hcc 
    controlUtility_treated = controlUtility_treatment
    controlUtility_falsePositiveHCC = controlUtility_masld
    controlUtility_death = 0.00
    controlUtility_cirrhosis = 0.00
    controlUtility_column = [controlUtility_masld, controlUtility_hcc, controlUtility_treatment, controlUtility_treated, controlUtility_falsePositiveHCC, controlUtility_death, controlUtility_cirrhosis]

    # Intervention Utility column
    interventionUtility_masld = controlUtility_masld
    interventionUtility_hcc = screen_hccDistribution_early * hccStages_costUtility['Control Utility'][0] + screen_hccDistribution_intermediate * hccStages_costUtility['Control Utility'][1] + screen_hccDistribution_late * hccStages_costUtility['Control Utility'][2]
    interventionUtility_treatment = interventionUtility_hcc
    interventionUtility_treated = interventionUtility_treatment 
    interventionUtility_falsePositiveHCC = controlUtility_falsePositiveHCC
    interventionUtility_death = controlUtility_death 
    interventionUtility_cirrhosis = 0.00
    interventionUtility_column = [interventionUtility_masld, interventionUtility_hcc, interventionUtility_treatment, interventionUtility_treated, interventionUtility_falsePositiveHCC, interventionUtility_death, interventionUtility_cirrhosis]

    data = {
        "Cost/Utility Matrix": ["MASLD", "HCC", "Treatment", "Treated", "False Positive HCC", "Death", "Cirrhosis"],
        "Control Cost": controlCost_column,
        "Intervention Cost": interventionCost_column,
        "Control Utility": controlUtility_column,
        "Intervention Utility": interventionUtility_column,
    }
    cost_utility_matrix = pd.DataFrame(data)

    return cost_utility_matrix

def generate_hccStages_cost_utility(hccStages_cost_utility_inputs):
    data = {
    "HCC Stages Cost/Utility": ["HCC Early", "HCC Intermediate", "HCC Late"],
    "Control Cost": [hccStages_cost_utility_inputs['controlCost_hccEarly'], hccStages_cost_utility_inputs['controlCost_hccIntermediate'], hccStages_cost_utility_inputs['controlCost_hccLate']],
    "Intervention Cost": [hccStages_cost_utility_inputs['controlCost_hccEarly'], hccStages_cost_utility_inputs['controlCost_hccIntermediate'], hccStages_cost_utility_inputs['controlCost_hccLate']],
    "Control Utility": [hccStages_cost_utility_inputs['controlUtility_hccEarly'], hccStages_cost_utility_inputs['controlUtility_hccIntermediate'], hccStages_cost_utility_inputs['controlUtility_hccLate']],
    "Intervention Utility": [hccStages_cost_utility_inputs['controlUtility_hccEarly'], hccStages_cost_utility_inputs['controlUtility_hccIntermediate'], hccStages_cost_utility_inputs['controlUtility_hccLate']],
}
    hccStages_cost_utility_matrix = pd.DataFrame(data)

    return hccStages_cost_utility_matrix

def create_final_input_dict(
    hccDistributionEarly, 
    hccDistributionIntermediate, 
    hccDistributionLate, 
    cirrhosisUnderdiagnosisRateInMasld_Rate, 
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD, 
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic, 
    masldIncidenceRates_falsePositiveHCC, 
    masldIncidenceRates_masldToCirrhosis,
    controlCost_hccEarly,
    controlCost_hccIntermediate,
    controlCost_hccLate,
    controlUtility_hccEarly,
    controlUtility_hccIntermediate,
    controlUtility_hccLate

    ):
    input_dict = {
        'hccDistributionEarly': hccDistributionEarly,
        'hccDistributionIntermediate': hccDistributionIntermediate,
        'hccDistributionLate': hccDistributionLate,
        'cirrhosisUnderdiagnosisRateInMasld_Rate': cirrhosisUnderdiagnosisRateInMasld_Rate,
        'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD': cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD,
        'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic': cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic,
        'masldIncidenceRates_falsePositiveHCC': masldIncidenceRates_falsePositiveHCC,
        'masldIncidenceRates_masldToCirrhosis': masldIncidenceRates_masldToCirrhosis,
        'controlCost_hccEarly': controlCost_hccEarly,
        'controlCost_hccIntermediate': controlCost_hccIntermediate,
        'controlCost_hccLate': controlCost_hccLate,
        'controlUtility_hccEarly': controlUtility_hccEarly,
        'controlUtility_hccIntermediate': controlUtility_hccIntermediate,
        'controlUtility_hccLate': controlUtility_hccLate

    }
    return input_dict
   
# This function is just to improve readability of the inputs
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data

def generate_simulation_inputs(generating_inputs):
    file_path = '~/Desktop/Inputs_CEA_v3.xlsx'
    df_readingsheet = read_spreadsheet_readingsheet(file_path)
    df_finalrewards = read_spreadsheet_finalrewards(file_path)
    
    # Alter based on the distribution of the input values
    inputs = generating_inputs
    final_transition = generate_final_transition(df_readingsheet, inputs)
    #print("Final Transition")
    print(f'{final_transition}\n')
    
    hccStages_cost_utility_matrix = generate_hccStages_cost_utility(inputs)
    #print("HCC Stages Cost/Utility")
    print(f'{hccStages_cost_utility_matrix}\n')

    final_cost_utility_matrix = generate_cost_utility(df_finalrewards, final_transition, hccStages_cost_utility_matrix, inputs)
    #print("Cost/Utility Matrix")
    print(f'{final_cost_utility_matrix}\n')

    output = (final_transition, hccStages_cost_utility_matrix, final_cost_utility_matrix)
    return output

if __name__ == "__main__":
    # Define ranges with increments for each parameter
    hccDistributionEarly_range = np.round(np.arange(0.4, 0.5, 0.01), 3)
    hccDistributionLate_range = np.round(np.arange(0.2, 0.4, 0.01), 3)
    cirrhosisUnderdiagnosisRateInMasld_Rate_range = np.round(np.arange(0.05, 0.07, 0.005), 3)
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD_range = np.round(np.arange(0.04, 0.05, 0.002), 3)
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic_range = np.round(np.arange(0.001, 0.002, 0.0005), 3)
    masldIncidenceRates_falsePositiveHCC_range = np.round(np.arange(0.09, 0.12, 0.01), 3)
    masldIncidenceRates_masldToCirrhosis_range = np.round(np.arange(0.005, 0.008, 0.001), 3)
    controlCost_hccEarly_range = np.round(np.arange(62000, 63000, 500), 3)
    controlCost_hccIntermediate_range = np.round(np.arange(116000, 118000, 1000), 3)
    controlCost_hccLate_range = np.round(np.arange(105000, 107000, 1000), 3)
    controlUtility_hccEarly_range = np.round(np.arange(0.35, 0.45, 0.05), 3)
    controlUtility_hccIntermediate_range = np.round(np.arange(0.25, 0.35, 0.05), 3)
    controlUtility_hccLate_range = np.round(np.arange(0.15, 0.25, 0.05), 3)

    start = time.time()
    inputs = []
    for hccDistributionEarly, hccDistributionLate in itertools.product(hccDistributionEarly_range, hccDistributionLate_range):
        hccDistributionIntermediate = round(1 - (hccDistributionEarly + hccDistributionLate), 3)
        
        if hccDistributionIntermediate < 0 or hccDistributionIntermediate > 1:
            continue

        for other_params in itertools.product(
            cirrhosisUnderdiagnosisRateInMasld_Rate_range,
            cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD_range,
            cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic_range,
            masldIncidenceRates_falsePositiveHCC_range,
            masldIncidenceRates_masldToCirrhosis_range,
            controlCost_hccEarly_range,
            controlCost_hccIntermediate_range,
            controlCost_hccLate_range,
            controlUtility_hccEarly_range,
            controlUtility_hccIntermediate_range,
            controlUtility_hccLate_range
        ):
            # Create an input dictionary for each combination
            input = create_final_input_dict(
                hccDistributionEarly, 
                hccDistributionIntermediate, 
                hccDistributionLate, 
                *other_params
            )
            inputs.append(input)

    stop = time.time() 
    elapsed_time = stop - start 
    print(f"Elapsed Time To Create All Input Dictionaries (~10.3 Million): {elapsed_time:.2f} seconds")
    
    random_input_combinations = random.sample(inputs, 5)
    start = time.time()
    for input in random_input_combinations:  # Displaying 5 random input combinations and their results
        print(json.dumps(convert_numpy_types(input), indent=4))
        generate_simulation_inputs(input)
    stop = time.time()
    elapsed_time = stop - start
    print(f"Elapsed Time To Generate 5 Matrix Combinations: {elapsed_time:.2f} seconds")
    # Given the time estimates... it would take 144 hours to compute matrices for all 10.368 million combinations (so... this isn't feasible)

''' 
These are all the variables that we can change in the input dictionary. 
Need to talk to Sovann about reasonable distributions for each to simulate over.

    hccDistributionEarly, 
    hccDistributionIntermediate, 
    hccDistributionLate, 
    cirrhosisUnderdiagnosisRateInMasld_Rate, 
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD, 
    cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic, 
    masldIncidenceRates_falsePositiveHCC, 
    masldIncidenceRates_masldToCirrhosis,
    controlCost_hccEarly,
    controlCost_hccIntermediate,
    controlCost_hccLate,
    controlUtility_hccEarly,
    controlUtility_hccIntermediate,
    controlUtility_hccLate
'''

'''
Example Output:

rayhanrazzak@DN2tf6mx ~ % /usr/local/bin/python3 /Users/rayhanrazzak/Downloads/generateInputs.py
{
    "hccDistributionEarly": 0.48,
    "hccDistributionIntermediate": 0.22,
    "hccDistributionLate": 0.3,
    "cirrhosisUnderdiagnosisRateInMasld_Rate": 0.06,
    "cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD": 0.046,
    "cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic": 0.002,
    "masldIncidenceRates_falsePositiveHCC": 0.09,
    "masldIncidenceRates_masldToCirrhosis": 0.006,
    "controlCost_hccEarly": 62500,
    "controlCost_hccIntermediate": 117000,
    "controlCost_hccLate": 106000,
    "controlUtility_hccEarly": 0.45,
    "controlUtility_hccIntermediate": 0.3,
    "controlUtility_hccLate": 0.2
}
       Control Matrix     MASLD      HCC  Treatment  Treated  False Positive HCC     Death  Cirrhosis
0               MASLD  0.888505  0.00464      0.000    0.000                0.09  0.010855      0.006
1                 HCC  0.000000  0.63200      0.158    0.000                0.00  0.210000      0.000
2           Treatment  0.000000  0.00000      0.000    1.000                0.00  0.000000      0.000
3             Treated  0.000000  0.63200      0.000    0.158                0.00  0.210000      0.000
4  False Positive HCC  1.000000  0.00000      0.000    0.000                0.00  0.000000      0.000
5               Death  0.000000  0.00000      0.000    0.000                0.00  1.000000      0.000
6           Cirrhosis  0.000000  0.00000      0.000    0.000                0.00  0.000000      1.000

  HCC Stages Cost/Utility  Control Cost  Intervention Cost  Control Utility  Intervention Utility
0               HCC Early         62500              62500             0.45                  0.45
1        HCC Intermediate        117000             117000             0.30                  0.30
2                HCC Late        106000             106000             0.20                  0.20

  Cost/Utility Matrix  Control Cost  Intervention Cost  Control Utility  Intervention Utility
0               MASLD        3537.0             3537.0            0.850               0.85000
1                 HCC       87540.0            76961.5            0.342               0.39235
2           Treatment       87540.0            76961.5            0.342               0.39235
3             Treated           0.0                0.0            0.342               0.39235
4  False Positive HCC        4091.0             4091.0            0.850               0.85000
5               Death           0.0                0.0            0.000               0.00000
6           Cirrhosis           0.0                0.0            0.000               0.00000

{
    "hccDistributionEarly": 0.49,
    "hccDistributionIntermediate": 0.18,
    "hccDistributionLate": 0.33,
    "cirrhosisUnderdiagnosisRateInMasld_Rate": 0.055,
    "cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD": 0.05,
    "cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic": 0.001,
    "masldIncidenceRates_falsePositiveHCC": 0.09,
    "masldIncidenceRates_masldToCirrhosis": 0.007,
    "controlCost_hccEarly": 62000,
    "controlCost_hccIntermediate": 117000,
    "controlCost_hccLate": 106000,
    "controlUtility_hccEarly": 0.35,
    "controlUtility_hccIntermediate": 0.3,
    "controlUtility_hccLate": 0.2
}
       Control Matrix     MASLD       HCC  Treatment  Treated  False Positive HCC     Death  Cirrhosis
0               MASLD  0.888584  0.003695     0.0000   0.0000                0.09  0.010721      0.007
1                 HCC  0.000000  0.609500     0.1595   0.0000                0.00  0.231000      0.000
2           Treatment  0.000000  0.000000     0.0000   1.0000                0.00  0.000000      0.000
3             Treated  0.000000  0.609500     0.0000   0.1595                0.00  0.231000      0.000
4  False Positive HCC  1.000000  0.000000     0.0000   0.0000                0.00  0.000000      0.000
5               Death  0.000000  0.000000     0.0000   0.0000                0.00  1.000000      0.000
6           Cirrhosis  0.000000  0.000000     0.0000   0.0000                0.00  0.000000      1.000

  HCC Stages Cost/Utility  Control Cost  Intervention Cost  Control Utility  Intervention Utility
0               HCC Early         62000              62000             0.35                  0.35
1        HCC Intermediate        117000             117000             0.30                  0.30
2                HCC Late        106000             106000             0.20                  0.20

  Cost/Utility Matrix  Control Cost  Intervention Cost  Control Utility  Intervention Utility
0               MASLD        3537.0             3537.0           0.8500               0.85000
1                 HCC       86420.0            76608.0           0.2915               0.32165
2           Treatment       86420.0            76608.0           0.2915               0.32165
3             Treated           0.0                0.0           0.2915               0.32165
4  False Positive HCC        4091.0             4091.0           0.8500               0.85000
5               Death           0.0                0.0           0.0000               0.00000
6           Cirrhosis           0.0                0.0           0.0000               0.00000

{
    "hccDistributionEarly": 0.48,
    "hccDistributionIntermediate": 0.13,
    "hccDistributionLate": 0.39,
    "cirrhosisUnderdiagnosisRateInMasld_Rate": 0.06,
    "cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD": 0.048,
    "cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic": 0.001,
    "masldIncidenceRates_falsePositiveHCC": 0.1,
    "masldIncidenceRates_masldToCirrhosis": 0.007,
    "controlCost_hccEarly": 62000,
    "controlCost_hccIntermediate": 116000,
    "controlCost_hccLate": 105000,
    "controlUtility_hccEarly": 0.4,
    "controlUtility_hccIntermediate": 0.3,
    "controlUtility_hccLate": 0.2
}
       Control Matrix     MASLD      HCC  Treatment  Treated  False Positive HCC     Death  Cirrhosis
0               MASLD  0.878325  0.00382      0.000    0.000                 0.1  0.010855      0.007
1                 HCC  0.000000  0.56000      0.167    0.000                 0.0  0.273000      0.000
2           Treatment  0.000000  0.00000      0.000    1.000                 0.0  0.000000      0.000
3             Treated  0.000000  0.56000      0.000    0.167                 0.0  0.273000      0.000
4  False Positive HCC  1.000000  0.00000      0.000    0.000                 0.0  0.000000      0.000
5               Death  0.000000  0.00000      0.000    0.000                 0.0  1.000000      0.000
6           Cirrhosis  0.000000  0.00000      0.000    0.000                 0.0  0.000000      1.000

  HCC Stages Cost/Utility  Control Cost  Intervention Cost  Control Utility  Intervention Utility
0               HCC Early         62000              62000              0.4                   0.4
1        HCC Intermediate        116000             116000              0.3                   0.3
2                HCC Late        105000             105000              0.2                   0.2

  Cost/Utility Matrix  Control Cost  Intervention Cost  Control Utility  Intervention Utility
0               MASLD        3537.0             3537.0            0.850                 0.850
1                 HCC       85790.0            76315.0            0.309                 0.357
2           Treatment       85790.0            76315.0            0.309                 0.357
3             Treated           0.0                0.0            0.309                 0.357
4  False Positive HCC        4091.0             4091.0            0.850                 0.850
5               Death           0.0                0.0            0.000                 0.000
6           Cirrhosis           0.0                0.0            0.000                 0.000

rayhanrazzak@DN2tf6mx ~ % 
'''