import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

prob_sensitivities = {
    (16, 1): [.242, .181, .302], 
    (63, 4): [.60, .45, .75], 
    (13, 1): [.01085, .00665, .01506],
    (18, 1): [.0004, .00004, .006],
    (17, 1): [.0225, .0208, .0243],
    (11, 1): [.15, .07, .27],
    (20, 1): [.00175, .00058, .00291], 
    (19, 1): [.017, .013, .021],
    (23, 2): [.533, .39975, .66625], 
    (39, 2): [.201, .15075, .25125],
    (53, 2): [.357, .268, .446],
    (26, 2): [.533, .39975, .66625], 
    (42, 2): [.369, .27675, .46125],
    (55, 2): [.632, .474, .790],
    (29, 2): [.389, .29175, .48625],
    (45, 2): [.792, .594, .990], 
    (57, 2): [.872, .654, 1], 
}


cost_sensitivities = {
    (21, 2): [363, 272, 454],
    (22, 2): [630, 473, 788], 
    (23, 2): [1050, 788, 1313],
    (0, 1): [4395, 3296, 5494], 
    (13, 1): [63255, 33875, 117193], 
    (16, 1): [47151, 43451, 50843], 
    (14, 1): [118753, 54307, 211436], 
    (17, 1): [51961, 44479, 60593],
    (15, 1): [105595, 45639, 144868], 
    (18, 1): [78547, 68621, 91278]
}


util_sensitivities = {
    (25, 2): [.85, .640, 1.0], 
    (10, 3): [.72, .62, .89], 
    (11, 3): [.69, .62, .82], 
    (12, 3): [.53, .2, .78],
}


prob_names = {
    (16, 1): "Proportion with undiagnosed cirrhosis",
    (63, 4): "Screening adherence rate",
    (13, 1): "Non-cirrhotic MASLD to cirrhosis (censored)",
    (18, 1): "Non-cirrhotic MASLD to HCC",
    (17, 1): "(Undiagnosed) cirrhosis with MASLD to HCC",
    (11, 1): "(undiagnosed cirrhosis) to false",
    (20, 1): "Non-cirrhotic MASLD to death",
    (19, 1): "(Undiagnosed) cirrhosis with MASLD to death",
    (23, 2): "Early-stage HCC to treatment",
    (39, 2): "Treated early-stage HCC to death",
    (53, 2): "Untreated early-stage HCC to death",
    (26, 2): "Intermediate-stage HCC to treatment",
    (42, 2): "Treated intermediate-stage HCC to death",
    (55, 2): "Untreated intermediate-stage HCC to death",
    (29, 2): "Late-stage HCC to treatment",
    (45, 2): "Treated late-stage HCC to death",
    (57, 2): "Untreated late-stage HCC to death"
}


cost_names = {
    (21, 2): "Semiannual US and AFP screening",
    (22, 2): "CT/MRI to confirm HCC diagnosis",
    (23, 2): "Repeat CT/MRI for false positive HCC",
    (0, 1): "Medical care of patients with MASLD (non-cirrhotic or with undiagnosed cirrhosis)",
    (13, 1): "Early stage HCC (annual costs) - Treated",
    (16, 1): "Early stage HCC (annual costs) - Untreated",
    (14, 1): "Intermediate stage HCC (annual costs) - Treated",
    (17, 1): "Intermediate stage HCC (annual costs) - Untreated",
    (15, 1): "Late stage HCC (annual costs) - Treated",
    (18, 1): "Late stage HCC (annual costs) - Untreated"
}


util_names = {
    (25, 2): "MASLD without cirrhosis",
    (10, 3): "Early stage HCC",
    (11, 3): "Intermediate stage HCC",
    (12, 3): "Late stage HCC"
}


def tornado(folder, num, file_name):
    data = {file: pd.read_csv(f'{folder}/{file}') for file in os.listdir(folder)}   
    legend = {0: 'Probabilities', 1: 'Costs', 2: 'Utilities'}
    sheets = pd.read_excel(file_name, sheet_name=None)
    nmb = -4625.944202081664  # Center point for the tornado plot
    low_outputs = []
    high_outputs = []
    var_names = []
    low_probs = []
    high_probs = []

    for var, df in data.items():
        var_tuple = eval(var.replace('.csv', ''))
        df = df.sort_values(by=df.columns[0])  
        low_val = np.percentile(df[df.columns[2]], 5)
        high_val = np.percentile(df[df.columns[2]], 95)
        low_outputs.append(low_val)   
        high_outputs.append(high_val)
        var_names.append(var_tuple)  
        low_probs.append(df.iloc[0][df.columns[0]])  
        high_probs.append(df.iloc[-1][df.columns[0]])  

    low_outputs = np.array(low_outputs)
    high_outputs = np.array(high_outputs)
    
    impact_ranges = abs(high_outputs - low_outputs)
    
    sorted_indices = np.argsort(impact_ranges)  # Negative for descending order
    sorted_vars = [var_names[i] for i in sorted_indices]  
    low_outputs = low_outputs[sorted_indices]
    high_outputs = high_outputs[sorted_indices]

    fig, ax = plt.subplots(figsize=(20, max(12, len(sorted_vars) * 0.8)))  
    y_positions = np.arange(len(sorted_vars))
    # Plot a single bar for each variable from low to high value
    bars = ax.barh(y_positions, high_outputs - low_outputs, left=low_outputs, color="#ff8888", alpha=0.7, height=0.6)
    # Add blue bars for lower bounds
    lower_bars = ax.barh(y_positions, low_outputs - nmb, left=nmb, color="#8888ff", alpha=0.7, height=0.6)

    # Get variable names and format labels with ranges
    if num == 0:
        base_labels = [prob_names.get(var, str(var)) for var in sorted_vars]
        range_labels = []
        for var in sorted_vars:
            low = prob_sensitivities[var][1]
            high = prob_sensitivities[var][2]
            range_labels.append(f"(varied from {low:.1%} to {high:.1%})")
    elif num == 1:
        base_labels = [cost_names.get(var, str(var)) for var in sorted_vars]
        range_labels = []
        for var in sorted_vars:
            low = int(cost_sensitivities[var][1])
            high = int(cost_sensitivities[var][2])
            range_labels.append(f"(varied from ${low:,} to ${high:,})")
    else:
        base_labels = [util_names.get(var, str(var)) for var in sorted_vars]
        range_labels = []
        for var in sorted_vars:
            low = util_sensitivities[var][1]
            high = util_sensitivities[var][2]
            range_labels.append(f"(varied from {low:.2f} to {high:.2f})")

    labels = []
    for base, range_ in zip(base_labels, range_labels):
        label = f"{base}\n{range_}"
        labels.append(label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    plt.subplots_adjust(left=0.5, right=0.95, bottom=0.1, top=0.95)
    ax.tick_params(axis='y', pad=10)
    ax.set_xlabel("Net Monetary Benefit ($)")
    ax.axvline(x=nmb, color='black', linestyle='-', alpha=0.2)
    def format_func(value, tick_number):
        return f'${int(value):,}'
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.margins(y=0.01)
    # Dynamically set x-axis limits based on data
    min_x = min(low_outputs.min(), high_outputs.min())
    max_x = max(low_outputs.max(), high_outputs.max())
    margin = 0.05 * (max_x - min_x)
    ax.set_xlim(min_x - margin, max_x + margin)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plt.savefig(f'/sailhome/malvlai/Cost-effectiveness/Results/One Way Sensitivity/Graphs/{legend[num]}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()


def two_way_plot(data_folder='/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Data'):
    control_data = [[.15, .05, .25], [.75, .55, .95]]
    intervention_data = [[.81, .6075, 1], [.11, .825, .1375]]
    
    control_df = pd.read_csv(f'{data_folder}/control.csv')
    intervention_df = pd.read_csv(f'{data_folder}/intervention.csv')
    
    scenarios = [
        ('Control', control_data, control_df),
        ('Intervention', intervention_data, intervention_df)
    ]
    
    for name, dist_data, df in scenarios:
        # ICER Contour Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.tricontourf(df['early_val'], df['late_val'], df['icer'], 
                                levels=20, cmap='RdYlBu')
        
        plt.colorbar(contour, label='ICER')
        
        ax.set_xlabel('Early Stage Probability')
        ax.set_ylabel('Late Stage Probability')
        ax.set_title(f'{name} Distribution\nTwo-Way Sensitivity Analysis of ICER')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.plot(dist_data[0][0], dist_data[1][0], 'ko', label='Baseline', markersize=10)
        ax.legend()
        
        plt.savefig(f'/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Graphs/{name} ICER.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        

        # Mean Death Contour Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.tricontourf(df['early_val'], df['late_val'], df['mean death'], 
                                levels=20, cmap='RdYlBu')
        
        plt.colorbar(contour, label='Mean Death From HCC')
        
        ax.set_xlabel('Early Stage Probability')
        ax.set_ylabel('Late Stage Probability')
        ax.set_title(f'{name} Distribution\nTwo-Way Sensitivity Analysis of Mean Death From HCC')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.plot(dist_data[0][0], dist_data[1][0], 'ko', label='Baseline', markersize=10)
        ax.legend()
        
        plt.savefig(f'/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Graphs/{name} Mean.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


        # Median Death Contour Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.tricontourf(df['early_val'], df['late_val'], df['median death'], 
                                levels=20, cmap='RdYlBu')
        
        plt.colorbar(contour, label='Median Death From HCC')
        
        ax.set_xlabel('Early Stage Probability')
        ax.set_ylabel('Late Stage Probability')
        ax.set_title(f'{name} Distribution\nTwo-Way Sensitivity Analysis of Median Death From HCC')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.plot(dist_data[0][0], dist_data[1][0], 'ko', label='Baseline', markersize=10)
        ax.legend()
        
        plt.savefig(f'/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Graphs/{name} Median.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 5 Year Survival Contour Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.tricontourf(df['early_val'], df['late_val'], df['5 year survival'], 
                                levels=20, cmap='RdYlBu')
        
        plt.colorbar(contour, label='5 Year Survival Rate From HCC')
        
        ax.set_xlabel('Early Stage Probability')
        ax.set_ylabel('Late Stage Probability')
        ax.set_title(f'{name} Distribution\nTwo-Way Sensitivity Analysis of 5 Year Survival From HCC')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.plot(dist_data[0][0], dist_data[1][0], 'ko', label='Baseline', markersize=10)
        ax.legend()
        
        plt.savefig(f'/sailhome/malvlai/Cost-effectiveness/Results/Two Way Sensitivity/Graphs/{name} Survival.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


    




def get_icer_color(icer_val):
    if 160000 <= icer_val < 170000:
        return '#fff5eb'  # yellow
    elif 170000 <= icer_val < 180000:
        return '#fdd0a2'  # green/gray
    elif 180000 <= icer_val < 190000:
        return '#fdae6b' # lilac
    elif 190000 <= icer_val < 200000:
        return '#e6550d' # blue
    else:
        return '#a63603'  # red


def hcc_distribution_plot(file, output):
    df = pd.read_csv(file)

    early_labels = ['5%', '10%', '15%', '20%', '25%']
    late_labels = ['55%', '65%', '75%', '85%', '95%']

    # Prepare data for 5x5 grid
    icer = df['icer'].values[:25].reshape(5, 5)
    median = df['median death'].values[:25].reshape(5, 5)
    mean = df['mean death'].values[:25].reshape(5, 5)
    survival = df['survival rate'].values[:25].reshape(5, 5)

    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(5):  # early (x)
        for j in range(5):  # late (y)
            color = get_icer_color(icer[i, j])
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor='white')
            ax.add_patch(rect)
            text = (f"ICER: {icer[i, j]:,.0f}\n"
                    f"Med: {median[i, j]:.1f}\n"
                    f"Mean: {mean[i, j]:.1f}\n"
                    f"Surv: {survival[i, j]:.2f}")
            ax.text(i + 0.5, j + 0.5, text, ha='center', va='center', fontsize=9, fontweight='bold' if icer[i, j] < 150000 else 'normal')

    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels(early_labels)
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_yticklabels(late_labels)
    ax.set_xlabel("Early Stage at Diagnosis (Control Group)")
    ax.set_ylabel("Late Stage at Diagnosis (Control Group)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color='#fff5eb', label='ICER $160,000-$170,000 / QALY'),
        mpatches.Patch(color='#fdd0a2', label='ICER $170,000-$180,000 / QALY'),
        mpatches.Patch(color='#fdae6b', label='ICER $180,000-$190,000 / QALY'), 
        mpatches.Patch(color='#e6550d', label='ICER $190,000-$200,000 / QALY'),
        mpatches.Patch(color='#a63603', label='ICER >$200,000 / QALY')
    ]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()
    plt.savefig(output + '/graph.png')


# tornado('/sailhome/malvlai/Cost-effectiveness/Results/One Way Sensitivity/Data/Probabilities', 0, "/sailhome/malvlai/Cost-effectiveness/Inputs/Inputs_CEA_v4_3.27.25.xlsx")
# tornado('/sailhome/malvlai/Cost-effectiveness/Results/One Way Sensitivity/Data/Costs', 1, "/sailhome/malvlai/Cost-effectiveness/Inputs/Inputs_CEA_v4_3.27.25.xlsx")
# tornado('/sailhome/malvlai/Cost-effectiveness/Results/One Way Sensitivity/Data/Utilities', 2, "/sailhome/malvlai/Cost-effectiveness/Inputs/Inputs_CEA_v4_3.27.25.xlsx")

# hcc_distribution_plot('/sailhome/malvlai/Cost-effectiveness/Results/HCC Distributions/Data/control.csv', '/sailhome/malvlai/Cost-effectiveness/Results/HCC Distributions/Graphs')
# two_way_plot()