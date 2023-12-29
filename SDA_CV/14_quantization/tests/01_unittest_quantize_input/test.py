import numpy as np

from solution import QuantizationParameters, quantize


def test_1():
    qp = QuantizationParameters(
        np.float64(0.00784314),
        np.int32(0),
        np.int32(-128),
        np.int32(127),
    )
    tensor_fp32 = np.arange(-8, 8, dtype=np.float32).reshape(4, 4) / 5
    tensor_int8 = quantize(tensor_fp32, qp)
    tensor_int8_ans = np.array(
        [
            [-128, -128, -128, -127],
            [-102, -76, -51, -25],
            [0, 25, 51, 76],
            [102, 127, 127, 127],
        ],
        dtype=np.int8,
    )
    assert tensor_int8.shape == tensor_fp32.shape, "Shape is wrong"
    assert tensor_int8.dtype == tensor_int8_ans.dtype, "Type is wrong"
    assert (
        tensor_int8.tolist() == tensor_int8_ans.tolist()
    ), "Quantized values are wrong"


def test_2():
    qp = QuantizationParameters(
        np.float64(0.00392156863),
        np.int32(-128),
        np.int32(-128),
        np.int32(127),
    )
    tensor_fp32 = np.arange(-2, 14, dtype=np.float32).reshape(4, 4) / 10
    tensor_int8 = quantize(tensor_fp32, qp)
    tensor_int8_ans = np.array(
        [
            [-128, -128, -128, -102],
            [-77, -52, -26, -1],
            [25, 50, 76, 101],
            [127, 127, 127, 127],
        ],
        dtype=np.int8,
    )
    assert tensor_int8.shape == tensor_fp32.shape, "Shape is wrong"
    assert tensor_int8.dtype == tensor_int8_ans.dtype, "Type is wrong"
    assert (
        tensor_int8.tolist() == tensor_int8_ans.tolist()
    ), "Quantized values are wrong"
