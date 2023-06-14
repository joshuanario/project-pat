import numpy as np
import scipy
import math
import statistics

def test1sample(asamples, target, aname):
    tteststats, ttestp = scipy.stats.ttest_1samp(asamples, popmean=target)
    if (tteststats == float("inf")) or (tteststats == float("-inf")) or math.isnan(ttestp):
        ttestp = None
        tteststats = None
    else:
        tteststats = float(tteststats)
    diffsamples = [i - target for i in asamples]
    _, wilcoxonp = scipy.stats.wilcoxon(diffsamples)
    anorm = testnormality(asamples)
    output = {
        'target': target,
        'a': aname,
        'n': len(asamples),
        'normality_test': anorm,
        'student_t_1samp(a,target)': {
            '(mean(a) - target)/se': tteststats,
            'p': ttestp,
        },
        'wilcoxon(a,target)': {
            'p': wilcoxonp,
        },
        'mean(a) - target': anorm['mean'] - target,
        'median(a) - target': anorm['median'] - target,
    }
    return output

def test2sample(asamples, bsamples, aname, bname):
    magic_test = abs((len(asamples) - len(bsamples)) / (len(asamples) + len(bsamples)))
    if (magic_test > 0.125):
        raise "invalid sampled values"
    output={}
    tteststats2sided, ttestp2sided = scipy.stats.ttest_ind(asamples, bsamples, alternative='two-sided', equal_var=True)
    if (tteststats2sided == float("inf")) or (tteststats2sided == float("-inf")) or math.isnan(ttestp2sided):
        ttestp2sided = None
        tteststats2sided = None
    else:
        ttestp2sided = float(ttestp2sided)
        tteststats2sided = float(tteststats2sided)
    tteststatsless, ttestpless = scipy.stats.ttest_ind(asamples, bsamples, alternative='less', equal_var=True)
    if (tteststatsless == float("inf")) or (tteststatsless == float("-inf")) or math.isnan(ttestpless):
        ttestpless = None
        tteststatsless = None
    else:
        ttestpless = float(ttestpless)
        tteststatsless = float(tteststatsless)
    tteststatsgreater, ttestpgreater = scipy.stats.ttest_ind(asamples, bsamples, alternative='less', equal_var=True)
    if (tteststatsgreater == float("inf")) or (tteststatsgreater == float("-inf")) or math.isnan(ttestpgreater):
        ttestpgreater = None
        tteststatsgreater = None
    else:
        ttestpgreater = float(ttestpgreater)
        tteststatsgreater = float(tteststatsgreater)
    anorm = testnormality(asamples)
    bnorm = testnormality(bsamples)
    levenep = None
    if (anorm['s'] != 0.0) and (bnorm['s'] != 0.0):
        _, levenep = scipy.stats.levene(asamples, bsamples)
        levenep = float(levenep)
    _, mannwhitneyp2sided = scipy.stats.mannwhitneyu(asamples, bsamples, alternative='two-sided')
    _, mannwhitneypless = scipy.stats.mannwhitneyu(asamples, bsamples, alternative='less')
    _, mannwhitneypgreater = scipy.stats.mannwhitneyu(asamples, bsamples, alternative='greater')
    output = {
        'a': aname,
        'b': bname,
        'n_a': len(asamples),
        'n_b': len(bsamples),
        'normality_test(a)': anorm,
        'normality_test(b)': bnorm,
        'levene(a,b)': {
            'p': levenep,
        },
        'student_t_ind(a,b)': {
            'a!=b': {
                '(mean(a) - mean(b))/se': tteststats2sided,
                'p': ttestp2sided,
            },
            'a<b': {
                '(mean(a) - mean(b))/se': tteststatsless,
                'p': ttestpless,
            },
            'a>b': {
                '(mean(a) - mean(b))/se': tteststatsgreater,
                'p': ttestpgreater,
            },
        },
        'mannwhitneyu(a,b)': {
            'a!=b': {
                'p': float(mannwhitneyp2sided),
            },
            'a<b': {
                'p': float(mannwhitneypless),
            },
            'a>b': {
                'p': float(mannwhitneypgreater),
            },
        },
        'mean(a) - mean(b)': anorm['mean'] - bnorm['mean'],
        'median(a) - median(b)': anorm['median'] - bnorm['median'],
    }
    return output

def testnormality(samples):
    anormalstat, anormaltest = scipy.stats.normaltest(samples)
    return {
        's^2 + k^2': float(anormalstat),
        'p': float(anormaltest),
        'mean': float(np.mean(samples)),
        's': float(np.std(samples)),
        'median': float(statistics.median(samples)),
    }