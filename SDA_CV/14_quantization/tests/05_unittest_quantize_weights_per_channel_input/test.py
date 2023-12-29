import numpy as np
import pytest

from solution import quantize_weights_per_channel


def test_1():
    w1_r = np.arange(-2, 7).reshape(1, 3, 3)
    max_abs_val1 = np.abs(w1_r).max()
    w1_r = np.float32(w1_r / max_abs_val1)
    w2_r = np.arange(-3, 1.5, 0.5).reshape(1, 3, 3)
    max_abs_val2 = np.abs(w2_r).max()
    w2_r = np.float32(w2_r / (2 * max_abs_val2))
    w_r = np.stack([w1_r, w2_r], axis=0)
    w_q, qps = quantize_weights_per_channel(w_r)
    w_q_ans = np.array(
        [
            [[[-42, -21, 0], [21, 42, 64], [85, 106, 127]]],
            [[[-127, -106, -85], [-64, -42, -21], [0, 21, 42]]],
        ],
        dtype=np.int8,
    )
    assert w_q.dtype == w_q_ans.dtype, "Type of the quantized tensor is wrong"
    assert w_q.shape == w_q_ans.shape, "Shape of the quantized tensor is wrong"
    assert w_q.tolist() == w_q_ans.tolist(), "Quantized values are wrong"
    assert type(qps) == type(
        []
    ), "Function should return a list with quantization parameters"
    assert len(qps) == 2, "Number of quantization parameters is wrong"
    assert (
        qps[0].q_min == -127 and qps[0].q_max == 127
    ), "Quantization range of channel 1 is wrong"
    assert (
        qps[1].q_min == -127 and qps[1].q_max == 127
    ), "Quantization range of channel 2 is wrong"
    assert qps[0].scale == pytest.approx(
        0.007874016,
        abs=0.0001,
    ), "Scale of channel 1 is wrong"
    assert qps[1].scale == pytest.approx(
        0.003937008,
        abs=0.0001,
    ), "Scale of channel 2 is wrong"
    assert qps[0].zero_point == 0, "Zero point of channel 1 is wrong"
    assert qps[1].zero_point == 0, "Zero point of channel 2 is wrong"
