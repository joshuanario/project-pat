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

def _print_histogram(a, b, title, aname, bname):
    joined = np.concatenate((a, b))
    bins = np.linspace(min(joined) * .97, max(joined) * 1.1, 100)
    plt.title(title)
    plt.hist(a, bins, alpha=0.5, label=aname)
    plt.hist(b, bins, alpha=0.5, label=bname)
    plt.savefig(os.path.join(os.getcwd(), '__test2sample_' + title + '.png'))
    plt.close('all')

class Test2SampleHypothesisTest(unittest.TestCase):

    def test_gaussianongaussian(self):
        a = _make_test_data(230.0)
        b = _make_test_data(200.0)
        _print_histogram(a, b, 'gaussianongaussian', 'mean=230.0', 'mean=200.0')
        result = benchmark.test2sample(a, b, 'Test benchmark.test2sample_gaussianongaussian_a', 'Test benchmark.test2sample_gaussianongaussian_b')
        print(result)
        
    def test_gaussianonbimodal(self):
        a = _make_test_data(230.0)
        b = _make_test_data(200.0, isGaussian=False)
        _print_histogram(a, b, 'gaussianonbimodal', 'mean=230.0', 'median=200.0')
        result = benchmark.test2sample(a, b, 'Test benchmark.test2sample_gaussianonbimodal_a', 'Test benchmark.test2sample_gaussianonbimodal_b' )
        print(result)

    def test_degenerate(self):
        a = [230.0 for i in range(100)]
        b = _make_test_data(200.0, isGaussian=False)
        _print_histogram(a, b, 'degenerate', 'Dirac delta at 230.0', 'median=200.0')
        result = benchmark.test2sample(a, b, 'Test benchmark.test2sample_degenerate_a', 'Test benchmark.test2sample_degenerate_b')
        print(result)

    def test_invalidsample(self):
        a = _make_test_data(200.0, isGaussian=False, point_count=200)
        b = _make_test_data(200.0, isGaussian=False, point_count=500)
        self.assertRaises(TypeError, benchmark.test2sample, a, b, 'Test benchmark.test2sample_degenerate_a', 'Test benchmark.test2sample_degenerate_b')