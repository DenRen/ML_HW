import numpy as np

from solution import multiply_by_quantized_multiplier


def get_approx_multiplier(n: np.int32, m0: np.int32) -> np.float64:
    res = 0.0
    m0_bin = np.binary_repr(m0, 32)
    for i in range(0, 32):
        res += np.int32(m0_bin[i]) * 2 ** (-i)
    res = res * 2 ** int(-n)
    return res


def run(accum, n, m0, ans):
    accum = np.int32(accum)
    n = np.int32(n)
    m0 = np.int32(m0)
    ans = np.int32(ans)
    res = multiply_by_quantized_multiplier(accum, n, m0)
    assert res.dtype == np.int32, "Type is wrong"
    assert res == ans, "Multiplication result is wrong"


def test_1():
    run(accum=-52426, n=0, m0=1284127975, ans=-31349)


def test_2():
    run(accum=0, n=0, m0=1284127975, ans=0)


def test_3():
    run(accum=-21, n=2, m0=1549677596, ans=-4)


def test_4():
    run(accum=82005, n=2, m0=1575809544, ans=15044)


def test_5():
    run(accum=-14, n=3, m0=1137962927, ans=-1)


def test_6():
    run(accum=55, n=11, m0=1137466543, ans=0)


def test_7():
    n_values = list(range(17))
    m0_values = [1073741824, 1288490188, 1503238552, 1717986916, 1932735280, 2147483647]
    accum_values = list(range(-100000, 100000, 1999))
    for n in n_values:
        for m0 in m0_values:
            for accum in accum_values:
                m_approx = get_approx_multiplier(np.int32(n), np.int32(m0))
                res = multiply_by_quantized_multiplier(
                    np.int32(accum),
                    np.int32(n),
                    np.int32(m0),
                )
                ans = m_approx * accum
                assert res.dtype == np.int32, "Type is wrong"
                assert (
                    abs(res - ans) < 0.6
                ), f"Multiplication result is wrong: {res} is not close enough to {ans}. It should be {int(np.rint(ans))}"
