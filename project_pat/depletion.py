import numpy as np
import piecewise_regression
import itertools
import functools

def testchances(observed, timestamps, hi_pass_filter, label):
    return _ll(observed, timestamps, hi_pass_filter, label)

def _ll(observed, timestamps, hi_pass_filter, label):
    if (len(observed) != len(timestamps)):
        raise "invalid observed values"
    maxx = min(len(observed), len(timestamps))
    X = np.array(
        timestamps
        ).reshape(-1, 1)
    y = np.array(
        observed
        )
    uniform = np.linspace(0, maxx, maxx)

    pw_fit = piecewise_regression.Fit(X, y, n_breakpoints=1)
    fitted = pw_fit.yy
    der = np.diff(fitted)/np.diff(uniform)
    der[der < 0.0] = 0.0
    der = np.append(der, der[-1])

    pw_fit2 = piecewise_regression.Fit(uniform, der, n_breakpoints=1)
    norm_yy = np.array(pw_fit2.yy)
    norm_yy[norm_yy <= hi_pass_filter] = 0.0
    norm_yy[norm_yy > hi_pass_filter] = 1.0
    groups = itertools.groupby(norm_yy, lambda x: "1" if x > hi_pass_filter else "0" )
    states = []
    for k, g in groups:
        glist = list(g)
        states.append((np.mean(glist), sum(1 for i in glist)))
    depletions = list(filter(lambda x: x[0] > 0.0, states))
    max_s = functools.reduce(lambda x, y: max(x, y[1]), depletions, float("-inf"))
    most_probable = list(filter(lambda x: x[1] == max_s and x[0] > 0.0, depletions))[0]
    def rule_of_succession(s, n):
        return (s + 1.0)/(n + 2.0)
    return {
        'label': label,
        'depletion_state': {
            'chance': rule_of_succession(most_probable[1], most_probable[1]),
            'rate': most_probable[0],
        },
    }
