import pandas as pd

def read_spreadsheet(file_path):
    return pd.read_excel(file_path, sheet_name='FinalTransition')


def generate_control_matrix(spreadsheet):
    control_matrix = spreadsheet.iloc[:7, 1:9]
    control_matrix.columns = ['MASLD', 'HCC', 'Treatment', 'Treated', 'False Positive HCC', 'Death', 'Cirrhosis', 'Check']
    return control_matrix

def generate_MASLD_incidence(spreadsheet):
    MASLD_matrix = spreadsheet.iloc[10:14, :2]
    MASLD_matrix.columns = ['Description', 'Rate']
    MASLD_matrix = MASLD_matrix.set_index('Description').transpose()
    return MASLD_matrix

def generate_cirrhosis_MASLD_underdiagnosis_rate(spreadsheet):
    cirrhosis_underdiagnosis = spreadsheet.iloc[16:21, :2]
    cirrhosis_underdiagnosis.columns = ['Description', 'Rate']
    cirrhosis_underdiagnosis = cirrhosis_underdiagnosis.set_index('Description').transpose()
    return cirrhosis_underdiagnosis

def generate_hcc_incidence_(df):
    hcc_incidence_rates_matrices = {}

    stages = {
        'Early': [23, 24, 25],
        'Intermediate': [26, 27, 28],
        'Late': [29, 30, 31],
        'Weighted': [32, 33, 34]
    }

    for stage, rows in stages.items():
        treated_control = round(df.iloc[rows[0], 2] * 100) / 100  
        treated_screen = round(df.iloc[rows[0], 3] * 100) / 100   
        death_control = round(df.iloc[rows[1], 2] * 100) / 100    
        death_screen = round(df.iloc[rows[1], 3] * 100) / 100     
        untreated_control = round(df.iloc[rows[2], 2] * 100) / 100  
        untreated_screen = round(df.iloc[rows[2], 3] * 100)  / 100 

        stage_data = pd.DataFrame({
            'Treated': [treated_control, treated_screen],
            'Death': [death_control, death_screen],
            'Untreated': [untreated_control, untreated_screen]
        }, index=['Control', 'Screen'])

        hcc_incidence_rates_matrices[stage] = stage_data

    return hcc_incidence_rates_matrices

def generate_hcc_distribution(df):
    hcc_distribution_section = df.iloc[53:56, [1, 2]]
    hcc_distribution_section = hcc_distribution_section.replace({'%': ''}, regex=True).apply(pd.to_numeric)

    hcc_distribution_matrix = pd.DataFrame({
        'Control': hcc_distribution_section.iloc[:, 0].values,  
        'Screen': hcc_distribution_section.iloc[:, 1].values   
    }, index=['Early', 'Intermediate', 'Late'])

    return hcc_distribution_matrix


if __name__ == "__main__":
    file_path = '~/Desktop/Inputs_CEA_v3.xlsx'
    df = read_spreadsheet(file_path)
    
    control_matrix = generate_control_matrix(df)
    print(control_matrix)

    MASLD_incidence = generate_MASLD_incidence(df)
    print(MASLD_incidence)

    cirrhosis_MASLD_underdiagnosis = generate_cirrhosis_MASLD_underdiagnosis_rate(df)
    print(cirrhosis_MASLD_underdiagnosis)

    hcc_incidence = generate_hcc_incidence_(df)
    for stage, matrix in hcc_incidence.items():
        print(f"\nHCC Incidence Rates - {stage}:")
        print(matrix)

    hcc_distribution = generate_hcc_distribution(df)
    print(hcc_distribution)


'''Example Output:

rayhanrazzak@DN0a2595ca ~ % /usr/local/bin/python3 /Users/rayhanrazzak/Downloads/readFinalInputs.p
y
  MASLD     HCC Treatment Treated  False Positive HCC  Death  Cirrhosis  Check
0  0.76   0.066         0       0                 0.1  0.024       0.05    1.0
1     0  0.1275    0.2775       0                 0.0  0.595       0.00    1.0
2     0       0         0       1                 0.0      0       0.00    1.0
3     0  0.1275         0  0.2775                 0.0  0.595       0.00    1.0
4     1       0         0       0                 0.0      0       0.00    1.0
5     0       0         0       0                 0.0      1       0.00    1.0
6     0       0         0       0                 0.0      0       1.00    1.0
Description MASLD to HCC False Positive HCC MASLD to Death MALSD to Cirrhosis
Rate               0.066                0.1          0.024               0.05
Description Rate MALSD to HCC for UD  ... MALSD to Death for UD MALSD to Death for non-cirrhotic
Rate         0.2                0.05  ...                  0.04                             0.02

[1 rows x 5 columns]

HCC Incidence Rates - Early:
         Treated  Death  Untreated
Control     0.05    0.0       0.95
Screen      0.95    0.0       0.05

HCC Incidence Rates - Intermediate:
         Treated  Death  Untreated
Control     0.20    0.0       0.80
Screen      0.95    0.0       0.05

HCC Incidence Rates - Late:
         Treated  Death  Untreated
Control     0.30   0.70        0.0
Screen      0.95   0.05        0.0

HCC Incidence Rates - Weighted:
         Treated  Death  Untreated
Control     0.28   0.60       0.13
Screen      0.95   0.01       0.05
/Users/rayhanrazzak/Downloads/readFinalInputs.py:54: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  hcc_distribution_section = hcc_distribution_section.replace({'%': ''}, regex=True).apply(pd.to_numeric)
              Control  Screen
Early            0.05     0.6
Intermediate     0.10     0.3
Late             0.85     0.1
rayhanrazzak@DN0a2595ca ~ % 

'''