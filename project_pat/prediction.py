import numpy as np
import scipy
import math

def testmodel(observed, fitted, stimuli, label):
    return _gm(observed, fitted, stimuli, label)

def _gm(observed, fitted, stimuli, label):
    n = min(len(observed), len(fitted), len(stimuli))
    count_of_independent_variables = 1
    degrees_of_freedom = count_of_independent_variables
    residuals = []
    for index in range(n):
        r = observed[index] - fitted[index]
        residuals.append(r)
    cm = np.cov([
        residuals,
        stimuli
    ])
    cmentries = []
    for r, row in enumerate(cm):
        for c, cell in enumerate(row):
            if r == c:
                continue
            cmentries.append(cell)
    tteststats, ttestp = scipy.stats.ttest_1samp(cmentries, popmean=0.0)
    if (tteststats == float("inf")) or (tteststats == float("-inf")) or math.isnan(ttestp):
        ttestp = None
        tteststats = None
    else:
        tteststats = float(tteststats)
    _slope, _intercept, r_value, _p, _se = scipy.stats.linregress(stimuli, residuals)
    coefficient_of_determination = r_value * r_value
    chi2testvar = n * coefficient_of_determination
    chi2pvalue = scipy.stats.chi2.sf(chi2testvar, degrees_of_freedom)
    output={}
    output = {
        'label': label,
        'n': n,
        'independence_of_error': {
            'mean(c)': np.mean(cmentries),
            'std(c)': np.std(cmentries),
            'student_t_1samp(c,target=0.0)': {
                '(mean(a) - target)/se': tteststats,
                'p': ttestp,
            },
        },
        'homoscedasticity_of_error': {
            'breusch_pagan': {
                'chi_square_test_variable': chi2testvar,
                'chi_square_p_value': chi2pvalue,
            }
        },
        'expected_error': {
            'mean(r)': np.mean(residuals),
            'std(r)': np.std(residuals)
        }
    }
    return output