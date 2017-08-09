'''
Implementation of common hypothesis tests.

email: dat.nguyen at cantab.net
'''
import numpy as np
from scipy import stats


def chisq_test(observed, expected, ddof=0, yates_correction=False):
    diffs = []
    for i, obs in enumerate(observed):
        exp = expected[i]
        if yates_correction:
            diff = (np.absolute(obs - exp) - 0.5)**2 / exp
        else:
            diff = (obs - exp) ** 2 / exp
        diffs.append(diff)
        
    chisq = sum(diffs)
    k = len(observed)

    # exp_freq_sum = sum(expected)

    pvalue = 1 - stats.chi2.cdf(chisq, k-1-ddof)

    return chisq, pvalue

def welch_ttest(sample1, sample2):

    mean_diff = np.mean(sample1) - np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    var1 = np.var(sample1)
    var2 = np.var(sample2)

    p_var = var1/n1 + var2/n2

    t_stat = mean_diff/np.sqrt(p_var)

    ddof = (p_var**2) / ( ((var1/n1)**2)/(n1-1) + ((var2/n2)**2)/(n2-1) )

    pvalue = 1 - stats.t.cdf(t_stat, ddof)

    return t_stat, pvalue

def ttest_mean(sample, h0_value):
    sample_mean = np.mean(sample)
    sample_var = np.var(sample)
    n = len(sample)

    t_stat = sample_mean-h0_value / np.sqrt(sample_var/n)

    pvalue = 1 - stats.t.cdf(t_stat, n-1)

    return t_stat, pvalue

def ftest_variance(sample1, sample2):
    n1 = len(sample1)
    var1 = np.sum(sample1 - np.mean(sample1))/(n1 - 1)

    n2 = len(sample2)
    var2 = np.sum(sample2 - np.mean(sample2))/(n2 - 1)

    F_stat = var1/var2

    pvalue = 1 - stats.f.cdf(F_stat, n1-1, n2-1)

    return F_stat, pvalue

def prop_test(counts, nobs, h0_value=0, correction=True):

    # test for one or two proportions
    if len(counts) <= 2:

        assert len(counts) == len(nobs)

        proportions = counts / nobs

        if len(proportions)==1:
            z_stat = (proportions - h0_value)/np.sqrt(h0_value*(1 - h0_value) / nobs)

        if len(proportions)==2:
            combined_prop = sum(counts)/sum(nobs)

            z_stat = (proportions[1]-proportions[0] - h0_value)/np.sqrt(combined_prop*(1 - combined_prop)*(1/nobs[0] + 1/nobs[1]))

        pvalue = stats.norm.cdf(z_stat)

        return z_stat, pvalue

    # chi-square test
    # this replicates R's prop.test function
    if len(counts) > 2:
        array = np.array([counts, nobs-counts]).T.reshape(-1, 2)
        results = stats.chi2_contingency(array, correction=correction)

        chisq_stat = results[0]
        pvalue = results[1]

        return chisq_stat, pvalue











    