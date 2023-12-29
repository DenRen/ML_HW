import numpy as np
import pytest
import torch

from solution import MinMaxObserver


def run(observer, min_val, max_val):
    assert observer.min.dtype == np.float32
    assert observer.max.dtype == np.float32
    assert observer.min == pytest.approx(
        min_val, abs=0.0001
    ), "The min observed value is wrong"
    assert observer.max == pytest.approx(
        max_val, abs=0.0001
    ), "The max observed value is wrong"


def test_1():
    observer = MinMaxObserver()
    input_tensor = torch.zeros((1, 3, 512, 512))

    # input1
    input_tensor[0, 0, 124, 253] = -2.47
    observer(input_tensor)
    run(observer, -2.47, 0)

    # input2
    input_tensor[0, 1, 10, 3] = 3.78
    observer(input_tensor)
    run(observer, -2.47, 3.78)

    # input3
    input_tensor[0, 2, 20, 13] = -4.11
    observer(input_tensor)
    run(observer, -4.11, 3.78)

    # input4
    input_tensor[0, 2, 200, 130] = 5.79
    observer(input_tensor)
    run(observer, -4.11, 5.79)

    # input5
    input_tensor[0, 0, 320, 132] = -3
    input_tensor[0, 1, 470, 32] = 3
    observer(input_tensor)
    run(observer, -4.11, 5.79)


def test_2():
    observer = MinMaxObserver()
    input_tensor = torch.zeros((1, 3, 4, 4))
    for i in range(1000):
        if i == 500:
            input_tensor[0, 0, 0, 0] = -5.53
        elif i == 750:
            input_tensor[0, 0, 0, 0] = 7.39
        else:
            input_tensor[0, 0, 0, 0] = i / 1000 - 0.5
        observer(input_tensor)
    run(observer, -5.53, 7.39)
