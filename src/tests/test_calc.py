import numpy as np

from calc import angle


def test_angle():
    vertex = np.array([0, 0], dtype=np.float64)
    p1 = np.array([1, 0], dtype=np.float64)
    p2 = np.array([0, 1], dtype=np.float64)

    assert np.allclose(angle(vertex, p1, p2), np.pi / 2)
