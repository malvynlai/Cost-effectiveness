import pandas as pd
import matplotlib.pyplot as plt

def graphs(c_data, i_data):
    states = {
    0: 'MASLD',
    1: 'HCC',
    2: 'Untreated',
    3: 'Treated',
    4: 'False Positive HCC',
    5: 'Death',
    6: 'Cirrhosis'
    }
    excluded_states = {0, 5, 6}  
    c_percentages = {state: {tran_state: [0] for tran_state in states if tran_state != state} for state in states if state not in excluded_states}
    i_percentages = {state: {tran_state: [0] for tran_state in states if tran_state != state} for state in states if state not in {5, 6}}
    c_prev = pd.DataFrame(index=c_data.index, columns=c_data.columns)
    i_prev = pd.DataFrame(index=i_data.index, columns=i_data.columns)
    c_masld_trans = {state: [0] for state in states if state not in {0, 4}}
    
    for t in range(c_data.shape[1]):
        if t > 0:
            for graph_state, dict in c_percentages.items():
                total_in_state = (c_prev.iloc[:, t] == graph_state).sum()
                if total_in_state > 0:
                    for state in dict:
                        trans_count = ((c_data.iloc[:, t] == state) & 
                                     (c_prev.iloc[:, t] == graph_state)).sum()
                        percentage = (trans_count / total_in_state * 100)
                        dict[state].append(percentage)
                else:
                    for state in dict:
                        dict[state].append(0)
            for graph_state, dict in i_percentages.items():
                total_in_state = (i_prev.iloc[:, t] == graph_state).sum()
                if total_in_state > 0:
                    for state in dict:
                        trans_count = ((i_data.iloc[:, t] == state) & (i_prev.iloc[:, t] == graph_state)).sum()
                        percentage = (trans_count / total_in_state * 100)
                        dict[state].append(percentage)
                else:
                    for state in dict:
                        dict[state].append(0)
            total_in_state = ((c_prev.iloc[:, t] == 0) | (c_prev.iloc[:, t] == 4)).sum()
            if total_in_state > 0:
                for state in c_masld_trans:
                    trans_count = ((c_data.iloc[:, t] == state) & 
                                 ((c_prev.iloc[:, t] == 0) | (c_prev.iloc[:, t] == 4))).sum()
                    percentage = (trans_count / total_in_state * 100)
                    c_masld_trans[state].append(percentage)
            else:
                for state in c_masld_trans:
                    c_masld_trans[state].append(0)
            if t < c_data.shape[1] - 1:
                c_prev.iloc[:, t+1] = c_data.iloc[:, t]
                i_prev.iloc[:, t+1] = i_data.iloc[:, t]
            

    plt.figure(figsize=[10, 6])
    for state, percentage in c_masld_trans.items():
        plt.plot(range(c_data.shape[1]), percentage, marker='o', 
        linestyle='--', 
        linewidth=2, 
        alpha=0.7, 
        label=f'MASLD to {states[state]}')
    plt.title(f'MASLD Transition Probabilities for Control')
    plt.xlabel('Time Step')
    plt.ylabel('Percentage %')
    plt.legend()
    plt.grid()
    plt.savefig(f'/Users/malvynlai/Library/CloudStorage/Box-Box/Cost-effectiveness/Verifications/Base Case Graphs/MASLD Transition Rates')
    plt.show()
    plt.close()


    for cur_state, percentages in c_percentages.items(): 
        plt.figure(figsize=[10, 6])
        for state, percentage in percentages.items():
            plt.plot(range(c_data.shape[1]), percentage, marker='o', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.7, 
            label=f'{states[cur_state]} to {states[state]}')
        plt.title(f'{states[cur_state]} Transition Probabilities for Control')
        plt.xlabel('Time Step')
        plt.ylabel('Percentage %')
        plt.legend()
        plt.grid()
        plt.savefig(f'/Users/malvynlai/Library/CloudStorage/Box-Box/Cost-effectiveness/Verifications/Base Case Graphs/{states[cur_state]} Transition Rates for Control')
        plt.show()
        plt.close()


    for cur_state, percentages in i_percentages.items(): 
        plt.figure(figsize=[10, 6])
        for state, percentage in percentages.items():
            plt.plot(range(c_data.shape[1]), percentage, marker='o', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.7, 
            label=f'{states[cur_state]} to {states[state]}')
        plt.title(f'{states[cur_state]} Transition Probabilities for Intervention')
        plt.xlabel('Time Step')
        plt.ylabel('Percentage %')
        plt.legend()
        plt.grid()
        plt.savefig(f'/Users/malvynlai/Library/CloudStorage/Box-Box/Cost-effectiveness/Verifications/Base Case Graphs/{states[cur_state]} Transition Rates for Intervention')
        plt.show()
        plt.close()
        
