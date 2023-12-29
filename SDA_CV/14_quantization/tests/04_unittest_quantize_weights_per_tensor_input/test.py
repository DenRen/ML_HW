import numpy as np
import pytest

from solution import quantize_weights_per_tensor


def test_1():
    w_r = np.arange(-2, 7).reshape(1, 3, 3)
    max_abs_val = np.abs(w_r).max()
    w_r = np.float32(w_r / max_abs_val)
    w_q, qp = quantize_weights_per_tensor(w_r)
    w_q_ans = np.array(
        [[[-42, -21, 0], [21, 42, 64], [85, 106, 127]]],
        dtype=np.int8,
    )
    assert w_q.dtype == w_q_ans.dtype, "Type of the quantized tensor is wrong"
    assert w_q.shape == w_q_ans.shape, "Shape of the quantized tensor is wrong"
    assert qp.q_min == -127 and qp.q_max == 127, "Quantization range is wrong"
    assert w_q.tolist() == w_q_ans.tolist(), "Quantized values are wrong"
    assert qp.scale == pytest.approx(0.007874016, abs=0.00001), "Scale is wrong"
    assert qp.zero_point == 0, "Zero point is wrong"


def test_2():
    w_r = np.arange(-3, 1.5, 0.5).reshape(1, 3, 3)
    max_abs_val = np.abs(w_r).max()
    w_r = np.float32(w_r / (2 * max_abs_val))
    w_q, qp = quantize_weights_per_tensor(w_r)
    w_q_ans = np.array(
        [[[-127, -106, -85], [-64, -42, -21], [0, 21, 42]]],
        dtype=np.int8,
    )
    assert w_q.dtype == w_q_ans.dtype, "Type of the quantized tensor is wrong"
    assert w_q.shape == w_q_ans.shape, "Shape of the quantized tensor is wrong"
    assert qp.q_min == -127 and qp.q_max == 127, "Quantization range is wrong"
    assert w_q.tolist() == w_q_ans.tolist(), "Quantized values are wrong"
    assert qp.scale == pytest.approx(0.003937008, abs=0.00001), "Scale is wrong"
    assert qp.zero_point == 0, "Zero point is wrong"
