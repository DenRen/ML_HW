import numpy as np

from solution import quantize_bias


def test_1():
    biases = np.array(
        [
            -0.0022,
            0.0917,
            0.0057,
            -0.0430,
            0.0092,
            -0.0079,
            -0.0071,
            0.0250,
            -0.0727,
            -0.0261,
        ],
        dtype=np.float32,
    )
    scales_w = np.array(
        [
            0.0032,
            0.0034,
            0.0033,
            0.0031,
            0.0038,
            0.0066,
            0.0039,
            0.0042,
            0.0028,
            0.0042,
        ],
        dtype=np.float64,
    )
    scale_x = np.float64(0.0182)
    biases_quant = [
        quantize_bias(biases[i], scales_w[i], scale_x) for i in range(len(biases))
    ]
    ans = np.array(
        [-38, 1482, 95, -762, 133, -66, -100, 327, -1427, -341], dtype=np.int32
    )
    for bq in biases_quant:
        assert bq.dtype == np.int32, "Type of the quantized bias is wrong"
    assert biases_quant == ans.tolist(), "Quantized biases are wrong"


def test_2():
    biases = np.array(
        [
            -0.0022,
            0.0917,
            0.0057,
            -0.0430,
            0.0092,
            -0.0079,
            -0.0071,
            0.0250,
            -0.0727,
            -0.0261,
        ],
        dtype=np.float32,
    )
    scale_w = np.float64(0.0038)
    scale_x = np.float64(0.0182)
    biases_quant = [
        quantize_bias(biases[i], scale_w, scale_x) for i in range(len(biases))
    ]
    ans = np.array(
        [-32, 1326, 82, -622, 133, -114, -103, 361, -1051, -377],
        dtype=np.int32,
    )
    for bq in biases_quant:
        assert bq.dtype == np.int32, "Type of the quantized bias is wrong"
    assert biases_quant == ans.tolist(), "Quantized biases are wrong"
