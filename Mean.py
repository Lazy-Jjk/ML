import numpy as np
from scipy import stats

# Step 1: Static dataset
data = np.array([10, 12, 23, 23, 16, 23, 21, 16])

# Step 2: Calculate central tendency and dispersion
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True)[0][0]
variance = np.var(data)
std_dev = np.std(data)

# Step 3: Print results
print("Data:", data)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std_dev)