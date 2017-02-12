import numpy as np
from chainer import cuda


def _sqnorm(x):
    with cuda.get_device(x):
        x = x.ravel()
        if x.dtype == np.complex64 or x.dtype == np.complex128:
            return abs(x.dot(x.conj()))
        else:
            return float(x.dot(x))


class GradientDisplaying(object):

    """Optimizer hook function for gradient displaying.

    Args:
        threshold (float): L2 norm threshold.

    Attributes:
        threshold (float): L2 norm threshold of gradient norm.

    """
    name = 'GradientDisplaying'

    def __init__(self, single=False):
        self.single_grads = single
        self.call_history = []

    def __call__(self, opt):
        sum2 = 0
        for name, param in opt.target.namedparams():
            if self.single_grads:
                print(name)
                print(param.grad)
            else:
                sum2 += _sqnorm(param.grad)
        norm = np.sqrt(sum2)
        print(norm)
        self.call_history.append(norm)

    def avg_norm(self):
        mean_i = np.asarray(self.call_history).mean()
        self.call_history.clear()
        return mean_i


class HardGradientClipping(object):

    name = 'HardGradientClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for name, param in opt.target.namedparams():
            for g in np.nditer(param.grad, op_flags=['readwrite']):
                if abs(g) > self.threshold:
                    with cuda.get_device(g):
                        g[...] = self.threshold if g > 0 else -self.threshold
