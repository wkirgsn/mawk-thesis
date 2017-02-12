import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers
from mawk.regressor import Regressor
from chainer.optimizer import GradientClipping as GradientScaling
from mawk.hooks import GradientDisplaying, HardGradientClipping
from chainer.functions.loss import mean_squared_error as MSE
from chainer.functions.activation import lstm
from chainer.links.connection import linear
from chainer import variable


class LSTM_layer(chainer.link.Chain):

    """Fully-connected LSTM layer. Customized MAWK.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.
        c (chainer.Variable): Cell states of LSTM units.
        h (chainer.Variable): Output at the previous timestep.

    """
    def __init__(self, in_size, out_size,
                 init_w_upward=None, init_w_lateral=None,
                 forget_gate_bias=None):
        super().__init__(
            upward=linear.Linear(in_size, 4 * out_size,
                                 initialW=init_w_upward,
                                 initial_bias=forget_gate_bias),
            lateral=linear.Linear(out_size, 4 * out_size, nobias=True,
                                  initialW=init_w_lateral),
        )
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM_layer, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM_layer, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        """Resets the internal state.

        It sets None to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h


class LSTM_peep_layer(chainer.link.Chain):
    def __init__(self, out_size, in_size=None, Wdist=None, Wscale=None):
        if in_size is None:
            in_size = out_size
        super().__init__(
            W_fh=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_fc=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_ih=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_ic=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_oh=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_oc=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_ch=linear.Linear(out_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_fx=linear.Linear(in_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, in_size, out_size)),
            W_ix=linear.Linear(in_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, in_size, out_size)),
            W_ox=linear.Linear(in_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, in_size, out_size)),
            W_cx=linear.Linear(in_size, out_size,
                   initialW=get_init_W(Wdist, Wscale, in_size, out_size))
        )
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super().to_cpu()
        if self.h is not None:
            self.h.to_cpu()
        if self.c is not None:
            self.c.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h, c):
        assert isinstance(h, chainer.Variable)
        assert isinstance(c, chainer.Variable)
        h_ = h
        c_ = c
        if self.xp == np:
            h_.to_cpu()
            c_.to_cpu()
        else:
            h_.to_gpu()
            c_.to_gpu()
        self.h = h_
        self.c = c_

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        ft = self.W_fx(x)
        it = self.W_ix(x)
        ct = self.W_cx(x)
        ot = self.W_ox(x)

        if self.h is not None and self.c is not None:
            ft += self.W_fh(self.h) + self.W_fc(self.c)
            it += self.W_ih(self.h) + self.W_ic(self.c)
            ct += self.W_ch(self.h)
            ot += self.W_oh(self.h)
        ft = F.activation.sigmoid.sigmoid(ft)
        it = F.activation.sigmoid.sigmoid(it)
        ct = F.activation.tanh.tanh(ct)
        ot = F.activation.sigmoid.sigmoid(ot + self.W_oc(ct))

        c = it * ct
        if self.c is not None:
            c += ft * self.c
        self.c = c
        self.h = ot * F.activation.tanh.tanh(self.c)
        return self.h

    def get_state(self):
        return self.c


class GRU_layer(chainer.link.Chain):
    def __init__(self, out_size, in_size=None, Wdist=None, Wscale=None):
        if in_size is None:
            in_size = out_size
        super().__init__(
            W_r=linear.Linear(
                in_size, out_size,
                initialW=get_init_W(Wdist, Wscale, in_size, out_size)),
            U_r=linear.Linear(
                out_size, out_size,
                initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W_z=linear.Linear(
                in_size, out_size,
                initialW=get_init_W(Wdist, Wscale, in_size, out_size)),
            U_z=linear.Linear(
                out_size, out_size,
                initialW=get_init_W(Wdist, Wscale, out_size, out_size)),
            W=linear.Linear(
                in_size, out_size,
                initialW=get_init_W(Wdist, Wscale, in_size, out_size)),
            U=linear.Linear(
                out_size, out_size,
                initialW=get_init_W(Wdist, Wscale, out_size, out_size))
        )
        self.state_size = out_size
        self.h = None
        self.reset_state()

    def to_cpu(self):
        super().to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = F.activation.sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        z = F.activation.sigmoid.sigmoid(z)
        h_bar = F.activation.tanh.tanh(h_bar)

        h_new = z * h_bar
        if self.h is not None:
            h_new += (1 - z) * self.h
        self.h = h_new
        return self.h


class FNN_net(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(FNN_1hl, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        return self.l2(h1)


class LSTM_peep_net(chainer.Chain):
    def __init__(self, n_in, n_units, n_out, n_hl,
                 wdistribution, wscaleheuristic, train_mode=True):
        layers = {}
        for n in range(n_hl):
            if n == 0:
                fan_in = n_in
                fan_out = n_units
            else:
                fan_in = fan_out = n_units

            layers['l'+str(n+1)] =\
                LSTM_peep_layer(fan_out, fan_in,
                                Wdist=wdistribution, Wscale=wscaleheuristic)
        weightmatrix = get_init_W(wdistribution, wscaleheuristic,
                                  n_units, n_out)
        layers['l'+str(n_hl+1)] = L.Linear(n_units, n_out,
                                           initialW=weightmatrix)
        super().__init__(**layers)
        self.n_hl = n_hl
        self.train = train_mode

    def __call__(self, x):
        y = self.l1(F.dropout(x, train=self.train))
        for n in range(1, self.n_hl + 1):
            layer = vars(self)['l'+str(n+1)]
            y = layer(F.dropout(y, train=self.train))
        return y

    def reset_state(self):
        for n in range(self.n_hl):
            layer = vars(self)['l'+str(n+1)]
            layer.reset_state()


class GRU_net(chainer.Chain):
    def __init__(self, n_in, n_units, n_out, n_hl,
                 wdistribution, wscaleheuristic, train_mode=True):
        layers = {}
        for n in range(n_hl):
            if n == 0:
                fan_in = n_in
                fan_out = n_units
            else:
                fan_in = fan_out = n_units

            layers['l'+str(n+1)] =\
                GRU_layer(fan_out, fan_in,
                          Wdist=wdistribution, Wscale=wscaleheuristic)
        weightmatrix = get_init_W(wdistribution, wscaleheuristic,
                                  n_units, n_out)
        layers['l'+str(n_hl+1)] = L.Linear(n_units, n_out,
                                           initialW=weightmatrix)
        super().__init__(**layers)
        self.n_hl = n_hl
        self.train = train_mode

    def __call__(self, x):
        y = self.l1(F.dropout(x, train=self.train))
        for n in range(1, self.n_hl + 1):
            layer = vars(self)['l'+str(n+1)]
            y = layer(F.dropout(y, train=self.train))
        return y

    def reset_state(self):
        for n in range(self.n_hl):
            layer = vars(self)['l'+str(n+1)]
            layer.reset_state()


class LSTM_net(chainer.Chain):
    def __init__(self, n_in, n_units, n_out, n_hl,
                 wdistribution, wscaleheuristic, train_mode=True):
        layers = {}
        for n in range(n_hl):
            if n == 0:
                fan_in = n_in
                fan_out = n_units
            else:
                fan_in = fan_out = n_units
            weightmatrix_up = get_init_W(wdistribution, wscaleheuristic,
                                         fan_in, 4 * fan_out)
            weightmatrix_lat = get_init_W(wdistribution, wscaleheuristic,
                                          fan_out, 4 * fan_out)

            # init forget gates with large values!
            fgbias = np.zeros((4*fan_out), np.float32)
            fgbias[2::8] = 0.5
            fgbias[6::8] = -0.5
            forget_gate_bias = fgbias

            layers['l'+str(n+1)] =\
                LSTM_layer(fan_in, fan_out,
                           init_w_upward=weightmatrix_up,
                           init_w_lateral=weightmatrix_lat,
                           forget_gate_bias=forget_gate_bias)

        weightmatrix = get_init_W(wdistribution, wscaleheuristic,
                                  n_units, n_out)
        layers['l'+str(n_hl+1)] = L.Linear(n_units, n_out,
                                           initialW=weightmatrix)

        super().__init__(**layers)
        self.n_hl = n_hl
        self.train = train_mode

    def __call__(self, x):
        y = self.l1(F.dropout(x, train=self.train))
        for n in range(1, self.n_hl + 1):
            layer = vars(self)['l'+str(n+1)]
            y = layer(F.dropout(y, train=self.train))
        return y

    def reset_state(self):
        for n in range(self.n_hl):
            layer = vars(self)['l'+str(n+1)]
            layer.reset_state()

    def fit_state_to_smaller_input(self, size=None):
        if size:  # test. This changes first LSTM layer only
            self.l1.h = chainer.Variable(self.l1.h.data[:size, :],
                                         volatile=self.l1.h.volatile)
            self.l1.c = chainer.Variable(self.l1.c.data[:size, :],
                                         volatile=self.l1.c.volatile)
        else:
            self.reset_state()


def get_init_W(distribution, scale_heuristic, in_size, out_size):
    """Initialize a random weight matrix.

    Scaling Heuristics:
        Standard: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf.
        Normalized: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf.

    Args:
        distribution: 'uniform' or 'unit_normal'
        scale_heuristic: "normalized_init" or "standard_init"
        in_size (Int): Fan-In
        out_size (Int): Fan-Out

    Returns:

    """
    assert distribution in ('uniform', 'unit_normal'), \
        'Distribution has to be either "uniform" or "unit_normal"'
    assert scale_heuristic in ('normalized_init', 'standard_init'), \
        'scale_heuristic has to be either "normalized_init" or "standard_init"'

    if scale_heuristic == 'standard_init':
        wscale = ((10.0 - 0.1)*np.random.random_sample() + 0.1)/np.sqrt(
            float(in_size))
    elif scale_heuristic == 'normalized_init':
        wscale = np.sqrt(6.0 / float(in_size + out_size))
    else:
        raise ValueError()

    if distribution == 'uniform':
        initialW = wscale * np.random.uniform(size=(out_size, in_size))
    elif distribution == 'unit_normal':
        initialW = np.random.normal(
                0, wscale * np.sqrt(1. / in_size), (out_size, in_size))
    else:
        raise ValueError()

    return initialW


opt_dict = {'adam': [optimizers.Adam, 'alpha'],  # 1e-3
            'adagrad': [optimizers.AdaGrad, 'lr'],  # 1e-1
            'adadelta': [optimizers.AdaDelta, 'eps'],  # 1e-7
            'sgd': [optimizers.SGD, 'lr'],  # 1e-2
            'momentumsgd': [optimizers.MomentumSGD, 'lr'],  # 1e-3
            'nesterov': [optimizers.NesterovAG, 'lr'],  # 1e-4
            'rmsprop': [optimizers.RMSprop, 'lr']  # 1e-3
            }

arch_dict = {'fnn': FNN_net,
             'rnn': None,
             'lstm': LSTM_net,
             'lstm_peep': LSTM_peep_net,
             'lstm-g': None,
             'gru': GRU_net}

reg_dict = {'weightdecay': [chainer.optimizer.WeightDecay, 'rate'],
            'gradclipping': [chainer.optimizer.GradientClipping, 'threshold'],
            'gaussnoise': [chainer.optimizer.GaussianNoise, 'std'],
            #'dropout': None
            }


def get_model_and_optimizer(**specs):

    net = arch_dict[specs.get('architecture')]
    loss_name = specs.get('loss_name')
    optim_name = specs.get('optim_name')
    reg = specs.get('regularization')
    lr_init = specs.get('lr_init')

    modelspecs = {key: specs[key] for key in ['n_in', 'n_out', 'n_hl',
                                              'n_units', 'wdistribution',
                                              'wscaleheuristic']}
    if specs.get('architecture') == 'fnn':
        pass
        # modelspecs['fnn_flashback'] = fnn_flashback
    elif specs.get('architecture') in ['lstm', 'gru', 'lstm_peep']:
        modelspecs['train_mode'] = False  # specs['train_mode']

    # Loss function
    if loss_name == 'mse':
        lossfun = MSE.mean_squared_error
    else:
        raise ValueError('Unsupported loss function name: ' + loss_name)

    # Architecture
    model = Regressor(net(**modelspecs), lossfun=lossfun)

    # Setup optimizer
    if optim_name in opt_dict.keys():
        arg = {opt_dict[optim_name][1]: lr_init}
        optimizer = opt_dict[optim_name][0](**arg)
    else:
        raise ValueError('Unsupported optimizing technique: ' + optim_name)

    optimizer.setup(model)
    for k in reg_dict.keys():
        if k in reg:
            arg = {reg_dict[k][1]: np.float32(reg[reg.index(k)+1])}
            optimizer.add_hook(reg_dict[k][0](**arg))

    # optimizer.add_hook(GradientDisplaying(single=False))
    return model, optimizer
