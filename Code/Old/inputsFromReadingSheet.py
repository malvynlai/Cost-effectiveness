import pandas as pd

def read_spreadsheet(file_path):
    return pd.read_excel(file_path, sheet_name='ReadingSheet')

def generate_control_matrix(spreadsheet):
    control_matrix = spreadsheet.iloc[:7, 1:8]
    control_matrix.columns = ['MASLD', 'HCC', 'Treatment', 'Treated', 'False Positive HCC', 'Death', 'Cirrhosis']
    return control_matrix

def generate_MASLD_incidence(spreadsheet):
    MASLD_matrix = spreadsheet.iloc[:4, 9:11]
    MASLD_matrix.columns = ['Description', 'Rate']
    MASLD_matrix = MASLD_matrix.set_index('Description').transpose()
    return MASLD_matrix

def generate_cirrhosis_MASLD_underdiagnosis_rate(spreadsheet):
    cirrhosis_underdiagnosis = spreadsheet.iloc[:5, 12:14]
    cirrhosis_underdiagnosis.columns = ['Description', 'Rate']
    cirrhosis_underdiagnosis = cirrhosis_underdiagnosis.set_index('Description').transpose()
    return cirrhosis_underdiagnosis

def generate_hcc_outcomes_rates(df):
    hcc_outcomes_rates_matrices = {}

    stages = {
        'Early': [0,1,2],
        'Intermediate': [3, 4, 5],
        'Late': [6, 7, 8],
        'Weighted': [9, 10, 11]
    }

    for stage, rows in stages.items():
        treated_control = round(df.iloc[rows[0], 17] * 100) / 100  
        treated_screen = round(df.iloc[rows[0], 18] * 100) / 100   
        death_control = round(df.iloc[rows[1], 17] * 100) / 100    
        death_screen = round(df.iloc[rows[1], 18] * 100) / 100     
        untreated_control = round(df.iloc[rows[2], 17] * 100) / 100  
        untreated_screen = round(df.iloc[rows[2], 18] * 100)  / 100 

        stage_data = pd.DataFrame({
            'Treated': [treated_control, treated_screen],
            'Death': [death_control, death_screen],
            'Untreated': [untreated_control, untreated_screen]
        }, index=['Control', 'Screen'])

        hcc_outcomes_rates_matrices[stage] = stage_data

    return hcc_outcomes_rates_matrices

def generate_treated_outcomes_rates(df):
    treated_outcomes_rates_matrices = {}

    stages = {
        'After Early': [0,1,2],
        'After Intermediate': [3, 4, 5],
        'After Late': [6, 7, 8],
        'Weighted': [9, 10, 11]
    }

    for stage, rows in stages.items():
        treated_control = round(df.iloc[rows[0], 22] * 100) / 100  
        treated_screen = round(df.iloc[rows[0], 23] * 100) / 100   
        death_control = round(df.iloc[rows[1], 22] * 100) / 100    
        death_screen = round(df.iloc[rows[1], 23] * 100) / 100     
        recurrence_control = round(df.iloc[rows[2], 22] * 100) / 100  
        recurrence_screen = round(df.iloc[rows[2], 23] * 100)  / 100 

        stage_data = pd.DataFrame({
            'Treated': [treated_control, treated_screen],
            'Death': [death_control, death_screen],
            'Recurrence': [recurrence_control, recurrence_screen]
        }, index=['Control', 'Screen'])

        treated_outcomes_rates_matrices[stage] = stage_data

    return treated_outcomes_rates_matrices

def generate_hcc_distribution(spreadsheet):
    data = {
        'Control': [df.iloc[0, 26], df.iloc[1, 26], df.iloc[2, 26]],
        'Screen': [df.iloc[0, 27], df.iloc[1, 27], df.iloc[2, 27]]
    }

    index = ['Early', 'Intermediate', 'Late']
    hcc_distribution = pd.DataFrame(data, index=index)
    
    return hcc_distribution

def generate_cost_utility_matrix(df)):
    data = {
        'Control Cost': [df.iloc[0, 30], df.iloc[1, 30], df.iloc[2, 30], df.iloc[3, 30], df.iloc[4, 30], df.iloc[5, 30], df.iloc[6, 30]],
        'Intervention Cost': [df.iloc[0, 31], df.iloc[1, 31], df.iloc[2, 31], df.iloc[3, 31], df.iloc[4, 31], df.iloc[5, 31], df.iloc[6, 31]],
        'Control Utility': [df.iloc[0, 32], df.iloc[1, 32], df.iloc[2, 32], df.iloc[3, 32], df.iloc[4, 32], df.iloc[5, 32], df.iloc[6, 32]],
        'Intervention Utility': [df.iloc[0, 33], df.iloc[1, 33], df.iloc[2, 33], df.iloc[3, 33], df.iloc[4, 33], df.iloc[5, 33], df.iloc[6, 33]],
    }

    index = ['MASLD', 'HCC', 'Treatment', 'Treated', 'False Positive HCC', 'Death', 'Cirrhosis']
    
    cost_utility_matrix = pd.DataFrame(data, index=index)
    
    return cost_utility_matrix

def generate_hcc_stages_cost_utility_matrix(spreadsheet):
    data = {
        'Control Cost': [df.iloc[0, 36], df.iloc[1, 36], df.iloc[2, 36]],
        'Intervention Cost': [df.iloc[0, 37], df.iloc[1, 37], df.iloc[2, 37]],
        'Control Utility': [df.iloc[0, 38], df.iloc[1, 38], df.iloc[2, 38]],
        'Intervention Utility': [df.iloc[0, 39], df.iloc[1, 39], df.iloc[2, 39]]
    }

    index = ['HCC Early', 'HCC Intermediate', 'HCC Late']
    hcc_cost_utility_matrix = pd.DataFrame(data, index=index)
    
    return hcc_cost_utility_matrix

if __name__ == "__main__":
    file_path = '~/Desktop/Inputs_CEA_v3.xlsx'
    df = read_spreadsheet(file_path)
    
    control_matrix = generate_control_matrix(df)
    #print(control_matrix)

    MASLD_matrix = generate_MASLD_incidence(df)
    #print(MASLD_matrix)

    cirrhosis_MASLD_underdiagnosis = generate_cirrhosis_MASLD_underdiagnosis_rate(df)
    #print(cirrhosis_MASLD_underdiagnosis)

    hcc_outcomes = generate_hcc_outcomes_rates(df)
    '''
    for stage, matrix in hcc_outcomes.items():
        print(f"\nHCC Outcome Rates - {stage}:")
        print(matrix)
    '''

    treated_outcomes = generate_treated_outcomes_rates(df)
    '''
    for stage, matrix in treated_outcomes.items():
        print(f"\nTreated Outcome Rates - {stage}:")
        print(matrix)
    '''


    hcc_distributions = generate_hcc_distribution(df)
    #print(hcc_distributions)

    cost_utility_matrix = generate_cost_utility_matrix(df)
    #print(cost_utility_matrix)

    hcc_cost_utility_matrix = generate_hcc_stages_cost_utility_matrix(df)
    #print(hcc_cost_utility_matrix)


''' 

EXAMPLE OUTPUT: 

rayhanrazzak@DN0a25916e ~ % /usr/local/bin/python3 /Users/rayhanrazzak/Downloads/inputsFromReadingSheet.py
    MASLD       HCC  Treatment  Treated  False Positive HCC     Death  Cirrhosis
0  0.8796  0.003572    0.00000  0.00000                 0.1  0.010828      0.006
1  0.0000  0.618150    0.16275  0.00000                 0.0  0.219100      0.000
2  0.0000  0.000000    0.00000  1.00000                 0.0  0.000000      0.000
3  0.0000  0.618150    0.00000  0.16275                 0.0  0.219100      0.000
4  1.0000  0.000000    0.00000  0.00000                 0.0  0.000000      0.000
5  0.0000  0.000000    0.00000  0.00000                 0.0  1.000000      0.000
6  0.0000  0.000000    0.00000  0.00000                 0.0  0.000000      1.000
Description  MASLD to HCC  False Positive HCC  MASLD to Death  MALSD to Cirrhosis
Rate             0.003572                 0.1        0.010828               0.006
Description   Rate  MALSD to HCC for UD  MALSD to HCC for non-cirrhotic  MALSD to Death for UD  MALSD to Death for non-cirrhotic
Rate         0.059                0.043                          0.0011                  0.036                           0.00925

HCC Outcome Rates - Early:
         Treated  Death  Untreated
Control     0.05    0.0       0.95
Screen      0.05    0.0       0.95

HCC Outcome Rates - Intermediate:
         Treated  Death  Untreated
Control     0.20    0.0       0.80
Screen      0.05    0.0       0.95

HCC Outcome Rates - Late:
         Treated  Death  Untreated
Control     0.30    0.7       0.00
Screen      0.05    0.0       0.95

HCC Outcome Rates - Weighted:
         Treated  Death  Untreated
Control     0.16   0.22       0.62
Screen      0.05   0.00       0.95

Treated Outcome Rates - After Early:
         Treated  Death  Recurrence
Control     0.05    0.0        0.95
Screen      0.95    0.0        0.05

Treated Outcome Rates - After Intermediate:
         Treated  Death  Recurrence
Control     0.20    0.0        0.80
Screen      0.95    0.0        0.05

Treated Outcome Rates - After Late:
         Treated  Death  Recurrence
Control     0.30   0.70         0.0
Screen      0.95   0.05         0.0

Treated Outcome Rates - Weighted:
         Treated  Death  Recurrence
Control     0.16   0.22        0.62
Screen      0.95   0.01        0.04
              Control  Screen
Early           0.457   0.707
Intermediate    0.230   0.156
Late            0.313   0.137
                    Control Cost  Intervention Cost  Control Utility  Intervention Utility
MASLD                   3537.000           3537.000           0.8500                 0.850
HCC                    88448.443          76791.723           0.3144                 0.357
Treatment              88448.443          76791.723           0.3144                 0.357
Treated                    0.000              0.000           0.3144                 0.357
False Positive HCC      4091.000           4091.000           0.8500                 0.850
Death                      0.000              0.000           0.0000                 0.000
Cirrhosis                  0.000              0.000           0.0000                 0.000
                  Control Cost  Intervention Cost  Control Utility  Intervention Utility
HCC Early              62340.0            62340.0              0.4                   0.4
HCC Intermediate      116996.0           116996.0              0.3                   0.3
HCC Late              105591.0           105591.0              0.2                   0.2
rayhanrazzak@DN0a25916e ~ % 
'''