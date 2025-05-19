import pandas as pd
import matplotlib.pyplot as plt
import os



# This takes the states output matrix to create desired graphs
file_path = os.path.expanduser('~/Desktop/Rayhan.csv')


def viz(file_path):

    states = {
        0: 'MASLD',
        1: 'HCC',
        2: 'Treatment',
        3: 'Treated',
        4: 'False Positive HCC',
        5: 'Death',
        6: 'Cirrhosis'
    }
    output_path_1 = os.path.expanduser("~/Desktop/masld_transition_percentages.png")
    output_path_2 = os.path.expanduser("~/Desktop/state_percentages_comparison.png")
    data = pd.read_csv(file_path, header=None)
    masld_percentages = []
    transition_percentages = {state: [] for state in states if state != 0}

    # --- First Plot: Transition Percentages (Carried-Over) ---


    last_transition_state_per_patient = [None] * len(data)

    for t in range(data.shape[1]):
        masld_percentage = (data.iloc[:, t] == 0).mean() * 100
        masld_percentages.append(masld_percentage)
        
        current_transition_percentages = {state: 0 for state in states if state != 0}
        
        if t > 0:
            previous_step = data.iloc[:, t - 1]
            current_step = data.iloc[:, t]
            
            for patient_index in range(len(data)):
                if previous_step.iloc[patient_index] == 0 and current_step.iloc[patient_index] != 0:
                    new_state = current_step.iloc[patient_index]
                    last_transition_state_per_patient[patient_index] = new_state  
                    
                    current_transition_percentages[new_state] += (1 / len(data)) * 100

                elif last_transition_state_per_patient[patient_index] is not None:
                    carry_state = last_transition_state_per_patient[patient_index]
                    current_transition_percentages[carry_state] += (1 / len(data)) * 100

            for state, percentage in current_transition_percentages.items():
                transition_percentages[state].append(percentage)
        else:
            for state in transition_percentages:
                transition_percentages[state].append(0)

    # --- Second Plot: Percentage of Patients in Each State ---

    state_percentages = {state: [] for state in states}

    for t in range(data.shape[1]):
        for state, state_name in states.items():
            state_percentage = (data.iloc[:, t] == state).mean() * 100
            state_percentages[state].append(state_percentage)

    # --- Third Plot: Specific Focus on Treated and Death States ---

    treated_state = 3  
    death_state = 5

    treated_percentages = []
    death_percentages = []

    for t in range(data.shape[1]):
        treated_percentage = (data.iloc[:, t] == treated_state).mean() * 100
        treated_percentages.append(treated_percentage)
        
        death_percentage = (data.iloc[:, t] == death_state).mean() * 100
        death_percentages.append(death_percentage)

    # --- Plotting ---

    # Figure 1: Only the first graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(data.shape[1]), masld_percentages, marker='o', linestyle='-', label='MASLD Percentage', alpha=0.8)
    for state, percentages in transition_percentages.items():
        plt.plot(
            range(data.shape[1]), 
            percentages, 
            marker='o', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.7, 
            label=f'MASLD to {states[state]}'
        )
    plt.title("MASLD and MASLD Transition Percentages Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path_1)
    plt.show()

    # Figure 2: Side-by-side comparison of the second and third graphs
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(18, 8))

    # Second Plot: Percentage of Patients in Each State
    for state, percentages in state_percentages.items():
        ax2.plot(
            range(data.shape[1]), 
            percentages, 
            marker='o', 
            linestyle='-', 
            label=states[state]
        )
    ax2.set_title("Percentage of Patients in Each State Over Time")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend()
    ax2.grid(True)

    # Third Plot: Percentage of Patients in Treated and Death States
    ax3.plot(
        range(data.shape[1]), 
        treated_percentages, 
        marker='o', 
        linestyle='-', 
        color='red', 
        label='Treated State'
    )
    ax3.plot(
        range(data.shape[1]), 
        death_percentages, 
        marker='o', 
        linestyle='-', 
        color='brown', 
        label='Death State'
    )
    ax3.set_title("Percentage of Patients in Treated and Death States Over Time")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Percentage (%)")
    ax3.legend()
    ax3.grid(True)

    # Display both figures
    plt.tight_layout()
    plt.savefig(output_path_2)
    plt.show()
return none
