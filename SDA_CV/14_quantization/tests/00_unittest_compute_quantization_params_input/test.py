import numpy as np
import pytest

from solution import compute_quantization_params


def run(r_min, r_max, q_min, q_max, scale_ans, zp_ans):
    r_min = np.float32(r_min)
    r_max = np.float32(r_max)
    q_min = np.int32(q_min)
    q_max = np.int32(q_max)
    qp = compute_quantization_params(r_min, r_max, q_min, q_max)
    assert type(qp.scale) == np.float64, "The type of scale is wrong."
    assert type(qp.zero_point) == np.int32, "The type of zero point is wrong."
    assert qp.scale == pytest.approx(scale_ans, abs=0.00001), "The scale is wrong"
    assert qp.zero_point == zp_ans, "The zero_point is wrong."


def test_1():
    run(r_min=0, r_max=1, q_min=0, q_max=255, scale_ans=0.00392, zp_ans=0)


def test_2():
    run(r_min=0, r_max=1, q_min=-128, q_max=127, scale_ans=0.00392, zp_ans=-128)


def test_3():
    run(r_min=0, r_max=1, q_min=0, q_max=127, scale_ans=0.00787, zp_ans=0)


def test_4():
    run(r_min=-1, r_max=0, q_min=-128, q_max=127, scale_ans=0.00392, zp_ans=127)


def test_5():
    run(r_min=-1, r_max=1, q_min=-128, q_max=127, scale_ans=0.00784, zp_ans=0)


def test_6():
    run(r_min=-1, r_max=1, q_min=-127, q_max=127, scale_ans=0.00787, zp_ans=0)


def test_7():
    run(r_min=-0.5, r_max=2, q_min=-128, q_max=127, scale_ans=0.00980, zp_ans=-77)


def test_8():
    run(r_min=-2, r_max=0.5, q_min=-128, q_max=127, scale_ans=0.00980, zp_ans=76)


def test_9():
    run(r_min=-0.51, r_max=1.37, q_min=-128, q_max=127, scale_ans=0.00737, zp_ans=-59)


def test_10():
    run(r_min=-128, r_max=127, q_min=-128, q_max=127, scale_ans=1.0, zp_ans=0)


def test_11():
    run(r_min=0, r_max=255, q_min=-128, q_max=127, scale_ans=1.0, zp_ans=-128)
