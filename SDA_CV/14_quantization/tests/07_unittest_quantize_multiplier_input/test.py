import numpy as np
import pytest

from solution import quantize_multiplier


def get_approx_multiplier(n: np.int32, m0: np.int32) -> np.float64:
    res = 0.0
    m0_bin = np.binary_repr(m0, 32)
    for i in range(0, 32):
        res += np.int32(m0_bin[i]) * 2 ** (-i)
    res = res * 2 ** int(-n)
    return res


def test_1():
    n, m0 = quantize_multiplier(np.float64(0.5))
    assert n.dtype == np.int32, "Type of n is wrong"
    assert m0.dtype == np.int32, "Type of m0 is wrong"
    assert n == 0, "n is wrong"
    assert m0 == 1073741824, "M0 is wrong"


def test_2():
    n, m0 = quantize_multiplier(np.float64(0.00032))
    assert n.dtype == np.int32, "Type of n is wrong"
    assert m0.dtype == np.int32, "Type of m0 is wrong"
    assert n == 11, "n is wrong"
    assert m0 == 1407374884, "M0 is wrong"


def test_3():
    for i in range(10000):
        m = (i + 1) / 2000
        n, m0 = quantize_multiplier(np.float64(m))
        assert n.dtype == np.int32, "Type of n is wrong"
        assert m0.dtype == np.int32, "Type of m0 is wrong"
        assert (
            m0 >= 1073741824 and m0 <= 2147483647
        ), "M0 must be in the range [1073741824, 2147483647]"
        approx_value = get_approx_multiplier(n, m0)
        assert approx_value == pytest.approx(
            m,
            abs=0.0000001,
        ), "Approximate multiplier is too far from the original real multiplier"
