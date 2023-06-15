from project_pat import depletion
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt

def _make_test_data(seg_count, point_count, leaky=True):
    x = []
    y = []
    s = 0
    base = 0.0
    additive = 0.0
    for index in range(point_count):
        x.append(index + np.random.normal(0.0, 0.01))
    while s < seg_count:
        for i in range(int(len(x)/seg_count)):
            if leaky and (s%2 != 0):
                additive += np.random.normal(456.0, 1.0)
                y.append(base + additive)
            else:
                y.append(base + np.random.normal(0.0, 1.0))
        s += 1
        additive = 0
    while len(y) < point_count:
        y.append(additive)
    return np.array(x), np.array(y)

def _print_plot(x, y, name):
    plt.plot(x, y, label=name)
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), '__testll_' + name + '.png'))
    plt.close('all')


class TestLLAlgorithm(unittest.TestCase):
    def test_depletion_present(self):
        maxx = 60*8
        X, y = _make_test_data(4, maxx)
        _print_plot(X,y,'testchances.depletion_present_ObservedValues')
        result = depletion.testchances(X, y, 1.0, 'Test depletion.testchances.depletion_present')
        print(result)

    def test_depletion_not_present(self):
        maxx = 60*8
        X, y = _make_test_data(4, maxx, leaky=False)
        _print_plot(X,y,'testchances.depletion_not_present_ObservedValues')
        result = depletion.testchances(X, y, 1.0, 'Test depletion.testchances.depletion_not_present')
        print(result)

    def test_invalidsample(self):
        maxx = 60*8
        y2 = [i for i in range(2*maxx)]
        X, y = _make_test_data(4, maxx, leaky=False)
        self.assertRaises(TypeError, depletion.testchances, X, y2)