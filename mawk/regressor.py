from chainer import link
import chainer.functions as F


class Regressor(link.Chain):
    """

    This is an example of a chain that wraps another chain. It computes the
    loss based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If True, compute accuracy on the forward
            computation. The default value is True.

    """
    compute_other_losses = False

    def __init__(self, predictor, lossfun):
        super(Regressor, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.other_loss1 = None
        self.predictor = predictor

    def __call__(self, x, t):
        """Computes the loss value for an input and label pair.

        It also can compute other losses and stores it to the attribute.

        Args:
            x (~chainer.Variable): Input minibatch.
            t (~chainer.Variable): Corresponding groundtruth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        self.y = None
        self.loss = None
        self.other_loss1 = None

        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t)
        if self.compute_other_losses:
            pass  # todo: implement calculating additional losses
        return self.loss
