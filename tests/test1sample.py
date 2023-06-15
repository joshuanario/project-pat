from project_pat import benchmark
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt

def _make_test_data(aimpoint, isGaussian = True, point_count = 100):
    x = []
    spread = aimpoint / (230.0/11.0)
    for index in range(point_count):
        if isGaussian:
            x.append(np.random.normal(aimpoint, spread))
        else:
            mode = np.random.choice([
                np.random.normal(aimpoint - 2.6 * spread, 1.1 * spread), 
                np.random.normal(aimpoint + 2.5 * spread, 0.9 * spread)
            ],
            1)[0]
            x.append(mode)
    return np.array(x)

def _print_histogram(samples, name):
    bins = np.linspace(min(samples) * .97, max(samples) * 1.1, 100)
    plt.title(name)
    plt.hist(samples, bins)
    plt.savefig(os.path.join(os.getcwd(), '__test1sample_' + name + '.png'))
    plt.close('all')

class Test1SampleHypothesisTest(unittest.TestCase):

    def test_ontargetgaussian(self):
        y = _make_test_data(230.0)
        _print_histogram(y, 'gaussian')
        result = benchmark.test1sample(y, 230, 'Test benchmark.test1sample_ontargetgaussian')
        print(result)
        
    def test_ontargetbimodal(self):
        y = _make_test_data(20.0, isGaussian=False)
        _print_histogram(y, 'bimodal')
        result = benchmark.test1sample(y, 20.0, 'Test benchmark.test1sample_ontargetbimodal')
        print(result)


    def test_ontargetdegenerate(self):
        y = [230.0 for i in range(100)]
        _print_histogram(y, 'degenerate')
        result = benchmark.test1sample(y, 20.0, 'Test benchmark.test1sample_ontargetdegenerate')
        print(result)
