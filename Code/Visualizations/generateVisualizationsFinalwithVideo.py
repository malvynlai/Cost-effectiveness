import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 
import imageio 
from completeSimulation import completeRunAge

def generateTransitionVisualization(data):
    states = {
        0: 'MASLD',
        1: 'HCC',
        2: 'Treatment',
        3: 'Treated',
        4: 'False Positive HCC',
        5: 'Death',
        6: 'Cirrhosis'
    }

    # --- First Plot: Transition Percentages (Carried-Over) ---

    masld_percentages = []
    transition_percentages = {state: [] for state in states if state != 0}

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
    #plt.savefig(output_path_1)
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
    #plt.savefig(output_path_2)
    plt.show()


def run_simulation_with_varied_inputs_and_create_video(input_dict, hcc_array):
    variable_map = {
        1: 'cirrhosisUnderdiagnosisRateInMasld_Rate',
        2: 'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD',
        3: 'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic',
        4: 'masldIncidenceRates_falsePositiveHCC',
        5: 'masldIncidenceRates_masldToCirrhosis',
        6: 'HCC[0]',
        7: 'HCC[1]',
        8: 'HCC[2]'
    }

    print("Select a variable to change:")
    for key, name in variable_map.items():
        print(f"{key}: {name}")

    variable_choice = int(input("Enter the number corresponding to the variable you want to change: "))
    min_value = float(input("Enter the minimum value for the range: "))
    max_value = float(input("Enter the maximum value for the range: "))
    increment = float(input("Enter the increment value: "))

    value_range = np.arange(min_value, max_value + increment, increment)

    selected_variable = variable_map[variable_choice]
    image_files = []

    for value in value_range:
        modified_input_dict = input_dict.copy()
        modified_hcc_array = np.array(hcc_array)

        if selected_variable in modified_input_dict:
            modified_input_dict[selected_variable] = value
        elif selected_variable.startswith("HCC["):
            index = int(selected_variable[4:-1])
            modified_hcc_array[index] = value
        else:
            raise ValueError(f"Invalid selection: {selected_variable}")

        raw_data = completeRunAge("file_path_placeholder", modified_input_dict, modified_hcc_array)
        data = pd.DataFrame(raw_data)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        generateTransitionVisualization(data, axs)  

        filename = f"frame_{value:.3f}.png"
        plt.suptitle(f"{selected_variable} = {value:.3f}")
        plt.savefig(filename)
        image_files.append(filename)
        plt.close(fig)

    with imageio.get_writer('output_animation.mp4', fps=2) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in image_files:
        os.remove(filename)

    print("Video created as output_animation.mp4")


''' Example run: 

1. You will be prompted to choose with variable to change
2. You will be asked a minimum, maximum, and increment for that variable
3. It will generate a video showing the change in the graphs as that variable changes (and the other remain constant)
'''

input_dict = {
    'cirrhosisUnderdiagnosisRateInMasld_Rate': 0.059,
    'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD': 0.043,
    'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic': 0.0011,
    'masldIncidenceRates_falsePositiveHCC': 0.1,
    'masldIncidenceRates_masldToCirrhosis': 0.006
}
HCC = np.array([0.457, 0.23, 0.313])
run_simulation_with_varied_inputs_and_create_video(input_dict, HCC)
