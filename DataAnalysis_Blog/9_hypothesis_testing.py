import numpy as np
from scipy.stats import norm

sample = np.array([152.8, 148.9, 149.6, 158.2, 162.9, 139.6, 153.5, 161.3])

z = (np.mean(sample) - 152) / (10 / np.sqrt(8))
print(z)

p_value = 1 - norm.cdf(z)
print(p_value)

