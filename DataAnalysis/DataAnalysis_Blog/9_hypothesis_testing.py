import numpy as np
import pandas as pd

from scipy.stats import norm, chi2, t, ttest_ind, ttest_rel, f
import statsmodels.stats.proportion as proportion
import statsmodels.api as sm

# 예제1
sample = np.array([152.8, 148.9, 149.6, 158.2, 162.9, 139.6, 153.5, 161.3])

z = (np.mean(sample) - 152) / (10 / np.sqrt(8))
print(z)

p_value = 1 - norm.cdf(z)
print(p_value)


# 예제2
p0 = 0.15
n = 450
x = 30

proportion = x / n
print("Proportion:", proportion)

test_statistic = (proportion - p0) / ((p0 * (1 - p0) / n) ** 0.5)
print("Test Statistic:", test_statistic)

p_value = norm.cdf(test_statistic)
print("P-value:", p_value)


# 예제3
x = 30
n = 450
p0 = 0.15

count = x
nobs = n
value = p0

stat, pval = proportion.proportions_ztest(count, nobs, value, alternative='smaller')

print("Test Statistic:", stat)
print("P-value:", pval)


# 예제4
data = [4.6, 4.4, 4.6, 4.7, 4.5, 4.3, 4.1, 4.7, 4.5, 4.2]

variance = np.var(data, ddof=1)
print("Variance:", variance)

test_statistic = (len(data) - 1) * variance / 0.08
print("Test Statistic:", test_statistic)

p_value = chi2.cdf(test_statistic, df=len(data)-1)
print("P-value:", p_value)


# 예제5
h2003 = [139.4, 139.8, 137.2, 149.2, 151.3, 131.9, 141.5, 152.3]
h2013 = [142.5, 138.9, 139.6, 148.2, 152.8, 129.5, 143.5, 151.3]

mean_h2003 = np.mean(h2003)
sd_h2003 = np.std(h2003, ddof=1)

mean_h2013 = np.mean(h2013)
sd_h2013 = np.std(h2013, ddof=1)

print("Mean h2003:", mean_h2003)
print("SD h2003:", sd_h2003)
print("Mean h2013:", mean_h2013)
print("SD h2013:", sd_h2013)

test_statistic = (mean_h2003 - mean_h2013) / np.sqrt(2 * 10**2 / 8)
print("Test Statistic:", test_statistic)

pz = norm.cdf(test_statistic, 0, 1)
p_value = 2 * pz

print("P-value (one-sided):", pz)
print("P-value (two-sided):", p_value)


# 예제6
n1 = 8
n2 = 8

var_h2003 = np.var(h2003, ddof=1)
var_h2013 = np.var(h2013, ddof=1)
vp = ((n1 - 1) * var_h2003 + (n2 - 1) * var_h2013) / (n1 + n2 - 2)
print("Pooled Variance (vp):", vp)

mean_h2003 = np.mean(h2003)
mean_h2013 = np.mean(h2013)
test_statistic = (mean_h2003 - mean_h2013) / np.sqrt(vp / n1 + vp / n2)
print("Test Statistic:", test_statistic)

p_value = 2 * t.cdf(test_statistic, df=(n1 + n2 - 2))
print("P-value:", p_value)


# 예제7
t_statistic, p_value = ttest_ind(h2003, h2013, equal_var=True)

print("Test Statistic:", t_statistic)
print("P-value:", p_value)


# 예제8
p_old = [5.2, 4.7, 5.0, 5.3, 4.9, 4.5, 5.4, 5.0, 5.2, 4.8, 5.5, 4.8]
p_new = [4.6, 4.4, 4.6, 4.7, 4.5, 4.3, 4.1, 4.7, 4.5, 4.2]

mean_p_old = np.mean(p_old)
var_p_old = np.var(p_old, ddof=1)

mean_p_new = np.mean(p_new)
var_p_new = np.var(p_new, ddof=1)

print("Mean p_old:", mean_p_old)
print("Variance p_old:", var_p_old)

print("Mean p_new:", mean_p_new)
print("Variance p_new:", var_p_new)

t_statistic, p_value = ttest_ind(p_old, p_new, equal_var=False, alternative='greater')

print("Test Statistic:", t_statistic)
print("P-value:", p_value)


# 예제9
pre_test = [74, 66, 64, 60, 58, 72, 67, 78, 77, 79]
post_test = [89, 80, 76, 65, 54, 66, 84, 76, 86, 79]

diff = np.array(pre_test) - np.array(post_test)

mean_diff = np.mean(diff)
sd_diff = np.std(diff, ddof=1)
test_statistic = mean_diff / (sd_diff / np.sqrt(len(diff)))

print("Mean Difference:", mean_diff)
print("Standard Deviation of Differences:", sd_diff)
print("Test Statistic:", test_statistic)

p_value = t.cdf(test_statistic, df=(len(diff) - 1))
print("P-value:", p_value)


# 예제10
t_statistic, p_value = ttest_rel(pre_test, post_test, alternative='less')

print("Test Statistic:", t_statistic)
print("P-value:", p_value)


# 예제11
n = np.array([400, 300])
err = np.array([30, 20])
p = err / n
print("Proportions (p):", p)

pp = np.sum(err) / np.sum(n)
print("Pooled Proportion (pp):", pp)

test_statistic = (p[0] - p[1]) / np.sqrt(pp * (1 - pp) * np.sum(1 / n))
print("Test Statistic:", test_statistic)

p_value = 1 - norm.cdf(test_statistic, 0, 1)
print("P-value:", p_value)


test_statistic, p_value = sm.stats.proportions_ztest(err, n, alternative='larger')

print("Test Statistic:", test_statistic)
print("P-value:", p_value)


# 예제12
p_old = [5.2, 4.7, 5.0, 5.3, 4.9, 4.5, 5.4, 5.0, 5.2, 4.8, 5.5, 4.8]
p_new = [4.6, 4.4, 4.6, 4.7, 4.5, 4.3, 4.1, 4.7, 4.5, 4.2]

test_statistic = np.var(p_old, ddof=1) / np.var(p_new, ddof=1)
print("Test Statistic:", test_statistic)

p_value = 2 * (1 - f.cdf(test_statistic, len(p_old) - 1, len(p_new) - 1))
print("P-value:", p_value)


# 예제13
f_statistic = np.var(p_old, ddof=1) / np.var(p_new, ddof=1)
dof1 = len(p_old) - 1
dof2 = len(p_new) - 1
p_value = f.cdf(f_statistic, dof1, dof2)

p_value = 2 * min(p_value, 1 - p_value)

print("F Statistic:", f_statistic)
print("P-value:", p_value)