from project_pat import prediction
import numpy as np
from scipy.stats import linregress
import unittest
import os
import matplotlib.pyplot as plt

def _make_test_data(n, offset=0.0, independent_errors=False):
    x = []
    for index in range(n):
        x.append(index)
    observed = []
    for index, i in enumerate(x):
        o = np.random.normal(i, 1.0)
        # o = random.normal(i, abs(random.normal(i*i*index,1.0))) # heteroscedasticity is hard to test with one predictor
        observed.append(o)
    slope, intercept, r_value, _p, _se = linregress(x, observed)
    coefficient_of_determination = r_value * r_value
    fitted = []
    for index, i in enumerate(x):
        se = np.random.normal(0, 1.0)
        offset_expectation = 0 if independent_errors else -1.0*offset*i + se
        f = i * slope + intercept + offset_expectation
        fitted.append(f)
    return x, observed, fitted, coefficient_of_determination

def _print_plot(x, observed, fitted, coefficient_of_determination, name):
    n = min(len(observed), len(fitted))
    plt.plot(x, observed, label=name + '_Observed')
    plt.plot(x, [observed[i]-fitted[i] for i in range(n)], label=name + '_Residual ' + str(coefficient_of_determination))
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), '__testgm_' + name + '.png'))
    plt.close('all')

class TestGMAlgorithm(unittest.TestCase):

    def test_zero_expected_error(self):
        n = 100
        x = []
        x, observed, fitted, coefficient_of_determination = _make_test_data(n)
        _print_plot(x,observed,fitted,coefficient_of_determination,'prediction.testmodel_zero_expected_error')
        result = prediction.testmodel(observed, fitted, x, 'Test prediction.testmodel_zero_expected_error')
        print('coefficient_of_determination: ', coefficient_of_determination)
        print(result)

    def test_positive_offset(self):
        n = 100
        x = []
        x, observed, fitted, coefficient_of_determination = _make_test_data(n,offset=1.0)
        _print_plot(x,observed,fitted,coefficient_of_determination,'prediction.testmodel_positive_expected_error')
        result = prediction.testmodel(observed, fitted, x, 'Test prediction.testmodel_positive_expected_error')
        print('coefficient_of_determination: ', coefficient_of_determination)
        print(result)

    def test_negative_offset(self):
        n = 100
        x = []
        x, observed, fitted, coefficient_of_determination = _make_test_data(n,offset=-1.0)
        _print_plot(x,observed,fitted,coefficient_of_determination,'prediction.testmodel_negative_expected_error')
        result = prediction.testmodel(observed, fitted, x, 'Test prediction.testmodel_negative_expected_error')
        print('coefficient_of_determination: ', coefficient_of_determination)
        print(result)

    def test_zero_expected_error_independent(self):
        n = 100
        x = []
        x, observed, fitted, coefficient_of_determination = _make_test_data(n,independent_errors=True)
        _print_plot(x,observed,fitted,coefficient_of_determination,'prediction.testmodel_zero_expected_error_independent')
        result = prediction.testmodel(observed, fitted, x, 'Test prediction.testmodel_zero_expected_error_independent')
        print('coefficient_of_determination: ', coefficient_of_determination)
        print(result)

    def test_positive_offset_independent(self):
        n = 100
        x = []
        x, observed, fitted, coefficient_of_determination = _make_test_data(n,offset=1.0,independent_errors=True)
        _print_plot(x,observed,fitted,coefficient_of_determination,'prediction.testmodel_positive_expected_error_independent')
        result = prediction.testmodel(observed, fitted, x, 'Test prediction.testmodel_positive_expected_error_independent')
        print('coefficient_of_determination: ', coefficient_of_determination)
        print(result)

    def test_negative_offset_independent(self):
        n = 100
        x = []
        x, observed, fitted, coefficient_of_determination = _make_test_data(n,offset=-1.0,independent_errors=True)
        _print_plot(x,observed,fitted,coefficient_of_determination,'prediction.testmodel_negative_expected_error_independent')
        result = prediction.testmodel(observed, fitted, x, 'Test prediction.testmodel_negative_expected_error_independent')
        print('coefficient_of_determination: ', coefficient_of_determination)
        print(result)