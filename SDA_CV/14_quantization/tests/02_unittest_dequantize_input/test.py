import numpy as np

from solution import QuantizationParameters, dequantize


def test_1():
    qp = QuantizationParameters(
        np.float64(0.00784313725490196),
        np.int32(0),
        np.int32(-128),
        np.int32(127),
    )
    tensor_int8 = np.array(
        [
            [-128, -128, -128, -127],
            [-102, -76, -51, -25],
            [0, 25, 51, 76],
            [102, 127, 127, 127],
        ],
        dtype=np.int8,
    )
    tensor_fp32 = dequantize(tensor_int8, qp)
    tensor_fp32_ans = np.array(
        [
            [-1.0039216, -1.0039216, -1.0039216, -0.9960785],
            [-0.8000001, -0.59607846, -0.40000004, -0.19607845],
            [0.0, 0.19607845, 0.40000004, 0.59607846],
            [0.8000001, 0.9960785, 0.9960785, 0.9960785],
        ],
        dtype=np.float32,
    )
    assert tensor_fp32.shape == tensor_int8.shape, "Shape is wrong"
    # assert tensor_fp32.dtype == tensor_fp32_ans.dtype, 'Type is wrong'
    np.testing.assert_almost_equal(
        tensor_fp32,
        tensor_fp32_ans,
        decimal=5,
        err_msg="Dequantized values are wrong",
    )


def test_2():
    qp = QuantizationParameters(
        np.float64(0.00392156863),
        np.int32(-128),
        np.int32(-128),
        np.int32(127),
    )
    tensor_int8 = np.array(
        [
            [-128, -128, -128, -102],
            [-77, -52, -26, -1],
            [25, 50, 76, 101],
            [127, 127, 127, 127],
        ],
        dtype=np.int8,
    )
    tensor_fp32 = dequantize(tensor_int8, qp)
    tensor_fp32_ans = np.array(
        [
            [0, 0, 0, 0.10196079],
            [0.2, 0.29803923, 0.4, 0.49803922],
            [0.6, 0.69803923, 0.8, 0.8980392],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    assert tensor_fp32.shape == tensor_int8.shape, "Shape is wrong"
    # assert tensor_fp32.dtype == tensor_fp32_ans.dtype, 'Type is wrong'
    np.testing.assert_almost_equal(
        tensor_fp32,
        tensor_fp32_ans,
        decimal=5,
        err_msg="Dequantized values are wrong",
    )
