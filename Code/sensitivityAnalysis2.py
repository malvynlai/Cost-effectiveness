from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import copyfile
import pandas as pd
import numpy as np
import os
import uuid
from tqdm import tqdm
from Visualizations.sensFinal import 

# File and constants
original_file = "/sailhome/malvlai/Cost-effectiveness/Inputs/Inputs_CEA_v4_changed HCC distributions_JLcopy_3.19.25_single input.xlsx"
output_base = "/sailhome/malvlai/Cost-effectiveness/Results/One Way Sensitivity/Data"

input_dict = {
    'cirrhosisUnderdiagnosisRateInMasld_Rate': 0.059,
    'cirrhosisUnderdiagnosisRateInMasld_MasldToHccForUD': 0.043,
    'cirrhosisUnderdiagnosisRateInMasld_MasldToHccNonCirrhotic': 0.0011,
    'masldIncidenceRates_falsePositiveHCC': 0.1,
    'masldIncidenceRates_masldToCirrhosis': 0.006
}

HCC = np.array([0.457, 0.23, 0.313])

# Sensitivity dictionaries
prob_sensitivities = {
    (16, 1): [.242, .181, .302], 
    (63, 4): [.60, .45, .75], 
    (13, 1): [.01085, .00665, .01506],
    (18, 1): [.0004, .00004, .006],
    (17, 1): [.0225, .0208, .0243],
    (11, 1): [.15, .7, .27],
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
    (57, 2): [.872, .654, 1]
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
    (12, 3): [.53, .2, .78]
}

def run_param_group(args):
    loc, value_range, param_type, size = args
    import xlwings as xw
    import completeSimulationv2_ver as run

    mean, lower, upper = value_range
    rng = np.random.default_rng()

    if param_type in ['prob', 'util']:
        sigma = (upper - lower) / 4
        alpha = mean * ((mean * (1 - mean)) / sigma ** 2 - 1)
        beta = alpha * (1 - mean) / mean
        samples = rng.beta(alpha, beta, size)
        print(f"[{param_type.upper()}] {loc} — Generated {len(samples)} samples")
    elif param_type == 'cost':
        sigma = (upper - lower) / 4
        shape = (mean / sigma) ** 2
        scale = sigma ** 2 / mean
        samples = rng.gamma(shape, scale, size)
    else:
        raise ValueError("Invalid param_type")

    results = []
    for i, sample_val in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)} for {param_type} at {loc} = {sample_val:.4f}")

    for sample_val in samples:
        temp_file = f"/tmp/{uuid.uuid4()}.xlsx"
        copyfile(original_file, temp_file)

        app = xw.App(visible=False)
        wb = app.books.open(temp_file)
        sheet_name = 'FinalTransition-Control' if param_type == 'prob' else 'FinalRewards'
        sheet = wb.sheets[sheet_name]
        sheet.cells(loc[0] + 1, loc[1] + 1).value = sample_val
        wb.app.calculation = 'automatic'
        wb.app.calculate()
        wb.save()
        wb.close()
        app.quit()

        sheets = pd.read_excel(temp_file, sheet_name=None)
        s, ageVector, ctrl_util, ctrl_rew, _ = run.completeRunAge(sheets, input_dict, HCC, intervention=False)
        s, ageVector, int_util, int_rew, _ = run.completeRunAge(sheets, input_dict, HCC, intervention=True)

        icer = (np.mean(int_rew) - np.mean(ctrl_rew)) / (np.mean(int_util) - np.mean(ctrl_util))
        nmb = 100000 * (np.mean(int_util) - np.mean(ctrl_util)) - (np.mean(int_rew) - np.mean(ctrl_rew))
        results.append([sample_val, icer, nmb])
        os.remove(temp_file)

    df = pd.DataFrame(results, columns=['Value', 'ICER', 'NMB'])
    out_dir = 'Probabilities' if param_type == 'prob' else 'Costs' if param_type == 'cost' else 'Utilities'
    out_path = os.path.join(output_base, out_dir, f'{loc}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return loc

if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["prob", "cost", "util", "all"], default="all", help="Subset of sensitivity analysis to run")
    args = parser.parse_args()

    size = 1000  
    jobs = []

    if args.type in ("prob", "all"):
        for loc, v in prob_sensitivities.items():
            jobs.append((loc, v, 'prob', size))
    if args.type in ("cost", "all"):
        for loc, v in cost_sensitivities.items():
            jobs.append((loc, v, 'cost', size))
    if args.type in ("util", "all"):
        for loc, v in util_sensitivities.items():
            jobs.append((loc, v, 'util', size))

    print(f"Running sensitivity analysis on {len(jobs)} variables with {size} samples each ({args.type})")

    for job in tqdm(jobs, desc="Running variables"):
        run_param_group(job)


    print("Generating tornado plots")

    from Visualizations.sensFinal import tornado
    if args.type in ("prob", "all"):
        tornado(os.path.join(output_base, "Probabilities"), 0)
    if args.type in ("cost", "all"):
        tornado(os.path.join(output_base, "Costs"), 1)
    if args.type in ("util", "all"):
        tornado(os.path.join(output_base, "Utilities"), 2)
