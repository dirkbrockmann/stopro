import numpy as np
import stopro as p


def test_wiener_shapes_and_time_grid_consistency():
    res = p.wiener(T=1.0, dt=0.01, N=2, samples=3, gap=5)
    X = res["X"]
    t = res["t"]

    assert X.shape == (3, len(t), 2)
    assert np.isclose(t[0], 0.0)
    assert np.isclose(t[-1], 1.0)


def test_ou_shapes():
    res = p.ornstein_uhlenbeck(T=1.0, dt=0.01, N=4, samples=2, gap=2)
    X = res["X"]
    t = res["t"]

    assert X.shape == (2, len(t), 4)
    assert np.isclose(t[0], 0.0)
    assert np.isclose(t[-1], 1.0)