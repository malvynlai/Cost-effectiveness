import math

def rate_to_annual_probability(rate, t):
    return 1 - math.exp(-rate * t)

def cumulative_to_annual_probability(cumulative_prob, years):
    rate = -math.log(1 - cumulative_prob) / years
    return 1 - math.exp(-rate)


# Early Stage HCC:

# After Transplant
earlyStage_transplant_annualProb = cumulative_to_annual_probability(0.35, 5)
print("Annual probability from early stage hcc transplant:", earlyStage_transplant_annualProb)

# After Ablation
earlyStage_ablation_annualProb = cumulative_to_annual_probability(0.568, 5)
print("Annual probability from early stage hcc ablation:", earlyStage_ablation_annualProb)

# After Radiotherapy
earlyStage_radiotherapy_annualProb = cumulative_to_annual_probability(0.296, 3)
print("Annual probability from early stage hcc radiotherapy:", earlyStage_radiotherapy_annualProb)

# Intermediate Stage HCC:

# After Radiotherapy
intermediateStage_radiotherapy_annualProb = cumulative_to_annual_probability(0.37, 2)
print("Annual probability from intermediate stage hcc radiotherapy:", intermediateStage_radiotherapy_annualProb)

# After Untreated
intermediateStage_untreated_annualProb = cumulative_to_annual_probability(0.86, 5)
print("Annual probability from intermediate stage hcc untreated:", intermediateStage_untreated_annualProb)

#Late Stage HCC:

# Untreated
lateStage_untreated_annualProb = cumulative_to_annual_probability(0.96, 5)
print("Annual probability from late stage hcc untreated:", lateStage_untreated_annualProb)