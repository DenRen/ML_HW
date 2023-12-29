import numpy as np
from numpy import array
import pickle
import os

import common as com
import solution

p = 2
h, w = 6, 4
a = np.arange(h * w).reshape(h, w)
a[0, 3] = 10
a[1, 2] = 11

b = a.reshape(h // p, p, w // p, p)

print(a, '\n')
print(b, '\n')

s = b.max(axis=(-3, -1))
print(s)
print(f's.shape: {s.shape}\n')

t = b.transpose(0, 2, 1, 3)
print(f't:\n{t}\n')

m = s[..., None, None]
print(m)

max_eq = t == m
print(max_eq)
print(f'max_eq.shape: {max_eq.shape}')

g = np.arange((h // p) * (w // p)).reshape(h // p, w // p)
g += 1

g = g[..., None, None]
print(f'g.shape: {g.shape}')

print(f'g:\n{g}\n')

mul = max_eq * g
print(mul.shape)

print(mul.transpose(0, 2, 1, 3).reshape(h, w))

exit()

test_path = 'C:\\Users\\denre\\Documents\\GitHub\\ML_HW\\SDA_CV\\6_nn_impl\\tests\\09_unittest_pooling2d_input'
test_data = com.load_test_data('pooling2d_backward', test_path)[4]

layer, extra_info = com.init_layer(solution.Pooling2D, test_data)
with com.SubTest(
    f"layer.forward(inputs)",
    extra_info=extra_info
):
    layer.forward(test_data['inputs'])

# Test backward pass
extra_info["grad_outputs"] = test_data['grad_outputs']
# with com.SubTest(
#     f"layer.backward(grad_outputs)",
#     extra_info=extra_info
# ):
    
actual=layer.backward(test_data['grad_outputs'])
correct=test_data['grad_inputs']

n, d, oh, ow = correct.shape

neq = actual != correct
print(np.where(neq))
print(actual[neq])
print(correct[neq])

# print(correct.shape)
print(actual[0, 1, 19, 4])
# for i_n in range(n):
#     for i_d in range(d):
#         print(actual[i_n, i_d].shape)
        # eq = actual[i_n, i_d] == correct[i_n, i_d]
        # print(actual[eq])

# print(actual[neq])

    # com.assert_ndarray_equal(
    #     actual=layer.backward(test_data['grad_outputs']),
    #     correct=test_data['grad_inputs']
    # )

# Test, that parameter gradients were calculated correctly
# for k, grad_value in test_data.items():
#     if k.startswith('param_grad_'):
#         grad_name = k[len('param_grad_'):] + '_grad'
#         with SubTest(
#             f"layer.{grad_name}",
#             extra_info=extra_info
#         ):
#             assert_ndarray_equal(
#                 actual=getattr(layer, grad_name, None),
#                 correct=grad_value
#             )


exit()

n, d = 10, 15

h, w = 8, 4
p = 2

a = np.arange(n * d * h * w).reshape(n, d, h, w)
print(a)

def hor_sum(a, p) -> np.ndarray:
    n, d, h, w = a.shape
    return np.sum(a.reshape(n, d, w * h // p, p), axis=-1).reshape(n, d, h, w // p)

c = hor_sum(hor_sum(a, p).transpose(0, 1, 3, 2), p).transpose(0, 1, 3, 2)

print(f'\n{c}')
print(c.shape)

exit()
# import solution

inputs=array([[[[ 205.,  985., -846., -828., 1014.],
                [-290., -936., -226.,  433., -237.],
                [-277.,  182., -816., -532.,  129.],
                [-360.,   12.,  633., -154., -293.],
                [ 105.,  374.,  581.,  706., -327.]]]])
kernels=array([[[[  98., -431., 586.],
                 [-139., -173., 191.],
                 [-700., -46.,  725.]]]])

kernels=array([[[[725., -46.,  -700.],[191., -173., -139.],[586., -431.,   98.]]]])

ref = array([[[[ 512735., 1538988., -1606436.],
               [ -134061., -1002309., 308979.],
               [  160454.,   494016.,  -465022.]]]])



exit()
# a = np.arange(2 * 2).reshape(2, 2)
# b = 2 * np.arange(2 * 2).reshape(2, 2)

# print(a)
# print(b)

# ref_res = np.sum(np.sum(a * b))
# print(f'ref_res: {ref_res}')

# print(np.tensordot(a, b, axes=([0,1],[0,1])))

# exit()

n, d, ih, iw = 3, 3, 4, 5
inputs = np.arange(n * d * ih * iw).reshape(n, d, ih, iw)

c, kh, kw = 2, 4, 5
kernel = 0.1 * np.arange(c * d * kh * kw).reshape(c, d, kh, kw)

print(inputs)
print()
print(kernel)
print()

res = np.tensordot(inputs, kernel, axes=([-3, -2, -1], [-3, -2, -1]))


print(res)
print(res.shape)
