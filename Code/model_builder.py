from keras.constraints import NonNeg
from keras import Model
from keras.layers import Input, Dense, Conv1D, MaxPool1D, Dropout, Flatten
from theory_guided_layers import LinearModel, HBVModel


def build_trad_model(seq, nvars, ft, ks, ds, nout, act):
    inp = Input(shape=(seq, nvars))
    cnn = Conv1D(filters=ft, kernel_size=ks, activation=act, padding='causal')(inp)
    cnn = MaxPool1D(padding='same')(cnn)
    cnn = Dropout(0.3)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(ds, activation=act)(cnn)
    out = Dense(nout, activation='linear')(cnn)

    model = Model(inputs=inp, outputs=out)
    return model


def build_hybrid_model(nvars=6, nout=3, seq=80, ft=2, ks=11, ds=3, act='tanh', h='linear', mode='normal',
                       inter_vars='qg'):
    # possible values for h : (1) h = 'linear' (2) h = 'complete' (3) h = 'complete_nc'
    if h == 'complete_nc':
        kc_conv = None
        kc_denv = None
        kc_denp = None
    else:
        kc_conv = NonNeg()
        kc_denv = NonNeg()
        kc_denp = NonNeg()

    inp = Input(shape=(seq, nvars))
    x = LinearModel(units=nout)(inp) if h == 'linear' else HBVModel(units=nout, mode=mode, inter_vars=inter_vars)(inp)
    x = Conv1D(filters=ft, kernel_size=ks, padding='causal', activation=act, kernel_constraint=kc_conv)(x)
    x = MaxPool1D(padding='same')(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(units=ds, activation=act, kernel_constraint=kc_denv)(x)
    y = Dense(units=nout, activation='linear', kernel_constraint=kc_denp)(x)

    model = Model(inputs=inp, outputs=y)
    return model


if __name__ == '__main__':
    m = build_hybrid_model(20, 10, seq=80, h='linear')
    print(m.summary())
