import keras.backend as K
from keras.layers import Layer
from keras import initializers, constraints


class LinearModel(Layer):
    def __init__(self, units, **kwargs):
        # layers params
        self.units, self.state_size = units, units

        # the physical parameters of the complete model
        self.kv = None

        super(LinearModel, self).__init__(name='linear', **kwargs)

    def weight_adder(self, name=None, v_ini=0.5, v_min=0.0, v_max=1.0, isTrainable=True):
        w = self.add_weight(name=name, shape=(1, self.units),  #
                            initializer=initializers.Constant(value=v_ini),
                            constraint=constraints.min_max_norm(min_value=v_min, max_value=v_max),
                            trainable=isTrainable)
        return w

    def build(self, input_shape):
        self.kv = self.weight_adder('kv', 0.5, 0.0, 1.0)  # correction factor for ep (-)
        super(LinearModel, self).build(input_shape)

    @staticmethod
    def rescaler(kv):
        kv_ = (2.0 - 0.25) * kv + 0.25
        return kv_

    def water_balance(self, vi, ep):
        kv = self.rescaler(self.kv)
        ea = kv * ep
        r = vi - ea
        return [r, ea]

    def call(self, inputs):
        u = self.units

        # input variables
        vi = inputs[:, :, :u]
        ep = inputs[:, :, u:]

        _r, _ea = self.water_balance(vi, ep)
        return _r

    def compute_output_shape(self, input_shape):
        nx, ny, nz = input_shape[0], input_shape[1], self.units
        return nx, ny, nz

    def get_config(self):
        return {'units': self.units}


class HBVModel(Layer):
    def __init__(self, units, mode='normal', inter_vars='qg', **kwargs):
        # layers params
        self.inter_vars = inter_vars
        self.units, self.state_size, self.mode = units, units, mode

        # the physical parameters of the complete model
        self.kv, self.fc, self.lp, self.cf, self.bt = None, None, None, None, None
        self.lu, self.ka, self.ab, self.ag, self.kp = None, None, None, None, None

        super(HBVModel, self).__init__(name='hbv', **kwargs)

    def weight_adder(self, name=None, v_ini=0.5, v_min=0.0, v_max=1.0, isTrainable=True):
        w = self.add_weight(name=name, shape=(1, self.units),  #
                            initializer=initializers.Constant(value=v_ini),
                            constraint=constraints.min_max_norm(min_value=v_min, max_value=v_max),
                            trainable=isTrainable)
        return w

    def build(self, input_shape):
        self.kv = self.weight_adder('kv', 0.5, 0.0, 1.0)  # correction factor for ep (-)
        self.fc = self.weight_adder('fc', 0.2, 0.0, 1.0)  # field capacity or the maximum soil moisture storage (mm)
        self.lp = self.weight_adder('lp', 0.5, 0.0, 1.0)  # limit above which ea reaches its potential value (-)
        self.cf = self.weight_adder('cf', 0.1, 0.0, 1.0)  # maximum value for Capillary Flow (mm/day)
        self.bt = self.weight_adder('bt', 2.0, 1.0, 6.0)  # param. of power relationship to simulate indirect runoff (-)
        self.ka = self.weight_adder('ka', 0.5, 0.0, 1.0)  # recession coefficient for runoff from quick runoff (1/day)
        self.ab = self.weight_adder('ab', 0.5, 0.0, 1.0)  # measure for non-linearity of flow in quick runoff res. (-)
        self.lu = self.weight_adder('lu', 0.1, 0.0, 1.0)  #
        self.ag = self.weight_adder('ag', 0.5, 0.0, 1.0)  # coef. of rec. coef. for runoff from base flow reservoir (-)
        self.kp = self.weight_adder('kp', 0.5, 0.0, 1.0)  # const. perc. rate occurring when water is available (mm/day)

        super(HBVModel, self).build(input_shape)

    @staticmethod
    def rescaler(kv, fc, lp, cf, bt, ka, ab, lu, ag, kp):
        kv_ = (2.0 - 0.25) * kv + 0.25
        fc_ = (1000.0 - 50.0) * fc + 50.0
        lp_ = (1.0 - 0.001) * lp + 0.001
        cf_ = (3.0 - 0.01) * cf + 0.01  # 10.0
        bt_ = bt
        ka_ = (0.50 - 0.10) * ka + 0.10
        ab_ = (0.60 - 0.50) * ab + 0.50
        lu_ = (50.0 - 0.001) * lu + 0.001
        ag_ = (0.33 - 0.02) * ag + 0.02
        kp_ = (4.0 - 0.001) * kp + 0.001
        return kv_, fc_, lp_, cf_, bt_, ka_, ab_, lu_, ag_, kp_

    @staticmethod
    def soil_moisture_reservoir(vi, ep, sm, fc, lp, kv, cf, bt):
        # 1. compute the abundant soil water (also referred to as direct runoff, sdr)
        sdr = K.maximum(sm + vi - fc, 0.0)

        # 2. compute the net amount of water that infiltrates into the soil (inet)
        inet = vi - sdr

        # 3. compute the actual evaporation
        tm = lp * fc
        ea = K.switch(condition=K.less(sm - tm, 0.0), then_expression=ep * (sm / tm), else_expression=ep) * kv

        # 4. compute the capillary flow ca = cf * (1.0 - sm / fc)
        ca = K.maximum(cf * (1.0 - sm / fc), 0.0)
        # ca = cf * (1.0 - sm / fc)

        # 5. compute part of the infiltrating water that will run off through the soil layer (seepage).
        sp = inet * K.pow(sm / fc, bt)

        return [inet, sdr, ea, ca, sp]

    @staticmethod
    def quick_runoff_reservoir(su, ka, lu, ab, kp):
        # 1. compute the quick runoff qh
        qh = ka * K.maximum(su - lu, 0) + ka * ab * su

        # 2. compute the amount of percolation
        qp = kp * su

        return [qh, qp]

    @staticmethod
    def baseflow_reservoir(sg, ka, ab, ag):
        # 1. compute the baseflow qg
        qg = ka * ab * ag * sg

        return qg

    @staticmethod
    def constraint_state_vars(sm, su, sg):
        sm = K.maximum(sm, 1.0)
        su = K.maximum(su, 0.0)
        sg = K.maximum(sg, 0.0)

        return sm, su, sg

    def step_do(self, step_in, states):
        u, p = self.units, 2 * self.units

        sm = states[0][:, :u]  # Soil moisture reservoir (mm)
        su = states[0][:, u:p]  # Quick runoff reservoir (mm)
        sg = states[0][:, p:]  # Baseflow reservoir (mm)

        # Load the current input column
        vi = step_in[:, :u]
        ep = step_in[:, u:]

        # rescale parameters
        kv_, fc_, lp_, cf_, bt_, ka_, ab_, lu_, ag_, kp_ = self.rescaler(self.kv, self.fc, self.lp, self.cf, self.bt,
                                                                         self.ka, self.ab, self.lu, self.ag, self.kp)

        # results from soil moisture reservoir
        [_inet, _sdr, _ea, _ca, _sp] = self.soil_moisture_reservoir(vi, ep, sm, fc_, lp_, kv_, cf_, bt_)

        # results from the quick runoff reservoir
        [_qh, _qp] = self.quick_runoff_reservoir(su, ka_, lu_, ab_, kp_)

        # results from the baseflow reservoir
        _qg = self.baseflow_reservoir(sg, ka_, ab_, ag_)

        # Water balance equations
        _dsm = _inet + _ca - _ea - _sp
        _dsu = _sp - _ca - _qh - _qp
        _dsg = _qp - _qg

        # next values of state variables
        next_sm = sm + K.clip(_dsm, -1e5, 1e5)
        next_su = su + K.clip(_dsu, -1e5, 1e5)
        next_sg = sg + K.clip(_dsg, -1e5, 1e5)

        # constrain values of sm, su and sl to be greater or equal to zero
        next_sm, next_su, next_sg = self.constraint_state_vars(next_sm, next_su, next_sg)

        # concatenate
        step_out = K.concatenate([next_sm, next_su, next_sg], axis=1)
        return step_out, [step_out]

    def call(self, inputs):
        u, p = self.units, 2 * self.units

        # define the initial state variables at the beginning
        init_states = [K.zeros((K.shape(inputs)[0], 3 * u))]

        # recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, inputs, init_states)

        sm = outputs[:, :, :u]
        su = outputs[:, :, u:p]
        sg = outputs[:, :, p:]

        # compute final process variables
        vi = inputs[:, :, :u]
        ep = inputs[:, :, u:]

        # rescale parameters
        kv_, fc_, lp_, cf_, bt_, ka_, ab_, lu_, ag_, kp_ = self.rescaler(self.kv, self.fc, self.lp, self.cf, self.bt,
                                                                         self.ka, self.ab, self.lu, self.ag, self.kp)

        # results from soil moisture reservoir
        [_inet, _sdr, _ea, _ca, _sp] = self.soil_moisture_reservoir(vi, ep, sm, fc_, lp_, kv_, cf_, bt_)

        # results from the quick runoff reservoir
        [_qh, _qp] = self.quick_runoff_reservoir(su, ka_, lu_, ab_, kp_)

        # results from the baseflow reservoir
        _qg = self.baseflow_reservoir(sg, ka_, ab_, ag_)

        # total discharge
        _q = _sdr + _qh + _qg

        if self.mode == "normal":
            return sg  # or _qp or _sp (choose the optimal intermediate physical variable)
        elif self.mode == "analysis":
            if self.inter_vars == 'ea':
                return _ea
            elif self.inter_vars == 'ca':
                return _ca
            elif self.inter_vars == 'sp':
                return _sp
            elif self.inter_vars == 'qp':
                return _qp
            elif self.inter_vars == 'qg':
                return _qg
            elif self.inter_vars == 'sm':
                return sm
            elif self.inter_vars == 'su':
                return su
            else:
                return sg

    def compute_output_shape(self, input_shape):
        nx, ny, nz = input_shape[0], input_shape[1], self.units
        return nx, ny, nz

    def get_config(self):
        return {'units': self.units, 'mode': self.mode, 'kv': self.kv, 'fc': self.fc, 'lp': self.lp, 'cf': self.cf,
                'bt': self.bt, 'lu': self.lu, 'ka': self.ka, 'ab': self.ab, 'ag': self.ag, 'kp': self.kp}
