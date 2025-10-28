import numpy as np
from scipy import stats

# Step 1: Sample data
data = np.array([12, 15, 14, 16, 13, 15, 14, 17, 16, 15])

# Step 2: Hypothesized (population) mean
mu_0 = 14

# Step 3: Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(data, mu_0)

# Step 4: Print results
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Step 5: Decision (alpha = 0.05)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis (mean is significantly different).")
else:
    print("Fail to reject the null hypothesis (no significant difference).")
