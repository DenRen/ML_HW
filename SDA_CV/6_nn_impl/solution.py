from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            updater.inertia *= self.momentum
            updater.inertia += self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        
        return np.fmax(inputs, 0)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """

        return np.where(self.forward_inputs >= 0, grad_outputs, 0)


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """

        inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp = np.exp(inputs)
        sum = np.sum(exp, axis=1, keepdims=True)
        return np.exp(inputs - np.log(sum))

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        
        m = grad_outputs * self.forward_outputs
        s = np.sum(m, axis=1, keepdims=True)
        return m - s * self.forward_outputs


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        return inputs @ self.weights + self.biases

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """

        self.weights_grad = self.forward_inputs.T @ grad_outputs
        self.biases_grad = np.sum(grad_outputs, axis=0)

        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """

        is_true = y_gt == True
        m = np.sum(np.sum(y_gt[is_true] * np.log(eps + y_pred[is_true])))
        m /= y_gt.shape[0]
        return -np.array([m])

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        y_pred_min = 1 / 4.5036e+15

        is_true = y_gt == True
        g = np.zeros_like(y_gt)
        g[is_true] = (-1 / y_gt.shape[0]) / np.clip(y_pred[is_true], y_pred_min, None)
        return g


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # 1) Create a Model
    opt = SGD(lr=1e-2)
    loss = CategoricalCrossentropy()
    model = Model(loss, opt)

    # 2) Add layers to the model
    layers = [
        Dense(128, (28 * 28,)), ReLU(),
        Dense(256), ReLU(),
        Dense(10),
        Softmax()
    ]
    for layer in layers:
        model.add(layer)

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=16, epochs=7)

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    n, _, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape
    oh, ow = ih - (kh - 1) + 2 * padding, iw - (kw - 1) + 2 * padding

    kernels = kernels[:, :, ::-1, ::-1]

    if padding > 0:
        npad = ((0,0), (0,0), (padding, padding), (padding, padding))
        inputs = np.pad(inputs, pad_width=npad, constant_values=0)

    outputs = np.zeros((n, c, oh, ow), dtype=inputs.dtype)

    iy_begin = 0
    for i_y in range(oh):
        ix_begin = 0
        for i_x in range(ow):
            window = inputs[:, :, iy_begin:iy_begin+kh, ix_begin:ix_begin+kw]
            outputs[:, :, i_y, i_x] = np.tensordot(window, kernels,
                                                   axes=([-3, -2, -1], [-3, -2, -1]))
            ix_begin += 1
        iy_begin += 1

    return outputs


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        padding = (self.kernel_size - 1) // 2
        biases = self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        return convolve(inputs, self.kernels, padding) + biases

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        padding = (self.kernel_size - 1) // 2

        sim = lambda X: X[:, :, ::-1, ::-1]
        T = lambda X: X.transpose(1, 0, 2, 3)

        grad_outputs_T = T(grad_outputs)
        inputs_sim_T = T(sim(self.forward_inputs))

        self.biases_grad = np.sum(grad_outputs_T, axis=(-1, -2, -3))
        self.kernels_grad = T(convolve(inputs_sim_T, grad_outputs_T, padding))

        return convolve(grad_outputs, T(sim(self.kernels)), padding)


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """

        p = self.pool_size

        # First solution
        # def hor_func(X, func):
        #     n, d, h, w = X.shape
        #     s = func(X.reshape(n, d, w * h // p, p), axis=-1)
        #     return s.reshape(n, d, h, w // p)

        # T_in = lambda X: X.transpose(0, 1, 3, 2)
        # hor_sum = lambda X: hor_func(X, np.sum)
        # return T_in(hor_sum(T_in(hor_sum(inputs)))) * (1 / (p * p))

        n, d, h, w = inputs.shape
        quad_view = inputs.reshape(n, d, h // p, p, w // p, p)
        axis = (-3, -1)
        match self.pool_mode:
            case 'avg':
                return quad_view.sum(axis=axis) * (1 / (p * p))
            case 'max':
                quad_view = quad_view.transpose(0, 1, 2, 4, 3, 5)
                quad_view = quad_view.reshape(n, d, h // p, w // p, p * p)
                self.max_idxs = np.argmax(quad_view, axis=-1, keepdims=True)
                return np.take_along_axis(quad_view, self.max_idxs, axis=-1).squeeze(axis=-1)


    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        
        n, d, oh, ow = self.forward_inputs.shape
        p = self.pool_size
        match self.pool_mode:
            case 'avg':
                g = grad_outputs[:, :, :, :, np.newaxis, np.newaxis]
                ones = (1 / (p * p)) * np.ones((n, d, oh // p, ow // p, p, p))
                return (ones * g).transpose(0, 1, 2, 4, 3, 5).reshape(n, d, oh, ow)
            case 'max':
                g = grad_outputs[..., np.newaxis, np.newaxis]
                
                n, d, h, w = self.forward_inputs.shape
                res = np.zeros((n, d, h // p, w // p, p * p))
                np.put_along_axis(res, self.max_idxs, 1, axis=-1)
                res = res.reshape(n, d, h // p, w // p, p, p)
                return (res * g).transpose(0, 1, 2, 4, 3, 5).reshape(n, d, oh, ow)


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """

        Add2D = lambda X: X[..., np.newaxis, np.newaxis]

        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3))
            var = np.var(inputs, axis=(0, 2, 3))

            self.forward_inverse_std = Add2D(1 / np.sqrt(eps + var))
            self.forward_centered_inputs = inputs - Add2D(mean)
            self.forward_normalized_inputs = self.forward_centered_inputs * \
                                             self.forward_inverse_std
            normalized_inputs = self.forward_normalized_inputs

            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * mean

            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * var
        else:
            inverse_std = Add2D(1 / np.sqrt(eps + self.running_var))
            centered_inputs = inputs - Add2D(self.running_mean)
            normalized_inputs = centered_inputs * inverse_std

        return Add2D(self.gamma) * normalized_inputs + Add2D(self.beta)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        Add2D = lambda X: X[..., np.newaxis, np.newaxis]

        dL_dXbn = grad_outputs * Add2D(self.gamma)

        dL_dVar = -0.5 * np.power(self.forward_inverse_std, 3)
        dL_dVar *= Add2D((dL_dXbn * self.forward_centered_inputs).sum(axis=(0, 2, 3)))

        n, _, h, w = grad_outputs.shape
        coef = 2 / (n * h * w)

        dL_dM = -self.forward_inverse_std * Add2D(np.sum(dL_dXbn, axis=(0, 2, 3)))
        dL_dM -= coef * dL_dVar * Add2D(np.sum(self.forward_centered_inputs, axis=(0, 2, 3)))

        dL_dX = dL_dXbn * self.forward_inverse_std
        dL_dX += coef * dL_dVar * self.forward_centered_inputs
        dL_dX += dL_dM * (coef / 2)

        self.gamma_grad = (grad_outputs * self.forward_normalized_inputs).sum(axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        return dL_dX


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        n, d, h, w = inputs.shape
        return inputs.reshape(n, d * h * w)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        return grad_outputs.reshape(*self.forward_inputs.shape)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            dist = np.random.uniform(0, 1, inputs.size)
            self.forward_mask = dist.reshape(*inputs.shape) >= self.p
            return self.forward_mask * inputs
        else:
            return (1 - self.p) * inputs

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """

        return self.forward_mask * grad_outputs


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # 1) Create a Model
    opt = SGDMomentum(lr=1e-2, momentum=0.9)
    loss = CategoricalCrossentropy()
    model = Model(loss, opt)

    # 2) Add layers to the model
    input_shape = (3, 32, 32)
    layers = [
        Conv2D(8, 3, input_shape), ReLU(), BatchNorm(),
        Conv2D(8, 3), ReLU(), BatchNorm(),
	    Pooling2D(2, 'max'),

        Conv2D(16, 3), ReLU(), BatchNorm(),
        Conv2D(16, 3), ReLU(), BatchNorm(),
	    Pooling2D(2, 'max'),
        
        Conv2D(32, 3), ReLU(), BatchNorm(),
        Conv2D(32, 3), ReLU(), BatchNorm(),
	    Pooling2D(2, 'max'),

        Flatten(),

        Dropout(0.2), Dense(1024), ReLU(),
        Dropout(0.2), Dense(1024), ReLU(),

        Dense(10), Softmax()
    ]
    for layer in layers:
        model.add(layer)

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=32, epochs=10, x_valid=x_valid, y_valid=y_valid)

    return model

# ============================================================================
