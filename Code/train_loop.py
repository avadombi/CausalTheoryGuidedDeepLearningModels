import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras import callbacks
from load_data import load_data
from model_builder import build_trad_model, build_hybrid_model
from basics_functions import custom_loss, normalizer, denormalizer, save_metrics, save_plot_obs_sim, save_losses, \
    save_hyper_params, save_shap_values


def train_trad_model(seq, nvars, ft, ks, ds, nout, act, x_train, y_train, x_val, y_val, epochs, bs, lr):
    # define model structure
    model = build_trad_model(seq, nvars, ft, ks, ds, nout, act)
    print(model.summary())
    # training
    optimizer = Adam(learning_rate=lr, epsilon=10E-3)
    model.compile(loss=custom_loss, optimizer=optimizer)

    # early stopping
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                 verbose=0, patience=15, restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='min',
                                         min_delta=0.005, cooldown=0, min_lr=lr / 100)

    # fit network
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, verbose=2,
                        batch_size=bs, callbacks=[es, reduce])
    return model, history


def train_trad_model_kfold(seq, pt, ft, ks, ds, act, isTraining, toNorm, lr, epochs, bs, start='2011-01-01',
                           end='2021-04-07', cluster=None, save_base=None, toResample=False, freq='1D'):
    # load data
    input_path = '../../Data/Calibration/inputs.xlsx'
    heads_path = '../../Data/Calibration/reference_head.xlsx'

    t, names, total_x, total_y, train_x, train_y, test_x, test_y = load_data(input_path,
                                                                             seq=seq, pt=pt,
                                                                             head_path=heads_path,
                                                                             isTraining=isTraining,
                                                                             cluster=cluster,
                                                                             toResample=toResample,
                                                                             freq=freq)

    # split into cross-validation datasets
    k = 10
    kFold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1

    for idx_train, idx_val in kFold.split(X=train_x, y=train_y):
        print('Start training fold %d/%d' % (fold, k))
        save_path = save_base + str(fold) + '/'

        # get training and validation data
        x_train, y_train = train_x[idx_train], train_y[idx_train]
        x_val, y_val = train_x[idx_val], train_y[idx_val]

        # inputs and outputs dimensions
        inp_dim = x_train.shape[-1]
        out_dim = y_train.shape[-1]

        # normalize (min, max) or standardize
        size = int(inp_dim / 2)

        # inputs characteristics
        rReshape = np.reshape(x_train, (-1, x_train.shape[-1]))

        if toNorm:
            lb, ub = np.mean(y_train, axis=0), np.std(y_train, axis=0)
            rAvg, rStd = np.mean(rReshape, axis=0), np.std(rReshape, axis=0)
        else:
            lb, ub = np.min(y_train, axis=0), np.max(y_train, axis=0)
            rAvg, rStd = np.min(rReshape, axis=0), np.max(rReshape, axis=0)

        _normalizer, _denormalizer = normalizer(toNorm=toNorm), denormalizer(toNorm=toNorm)

        # normalize (or standardize) x_train and x_val, y_train, y_val
        x_train = _normalizer(x_train, rAvg, rStd)
        x_val = _normalizer(x_val, rAvg, rStd)

        y_train = _normalizer(y_train, lb, ub)
        y_val = _normalizer(y_val, lb, ub)

        # normalize (or standardize) x_test and y_test
        x_test = _normalizer(test_x, rAvg, rStd)
        y_test = _normalizer(test_y, lb, ub)

        # build the model architecture
        nvars = inp_dim
        nout = out_dim

        # train model
        model, history = train_trad_model(seq, nvars, ft, ks, ds, nout, act, x_train, y_train, x_val, y_val, epochs, bs,
                                          lr)

        # save data
        # 1. save the model
        save_model_path = save_path + 'model'
        model.save_weights(filepath=save_model_path)

        # 2. Hyper-params
        hyper_params_dict = {
            'seq': seq, 'nvars': nvars, 'ft': ft, 'ks': ks, 'ds': ds,
            'nout': nout, 'act': act, 'lb': lb, 'ub': ub, 'rAvg': rAvg,
            'rStd': rStd, 'toNorm': toNorm, 'names': names.values
        }

        save_hyper_params(hyper_params_dict, save_path)

        # 3. Save metrics
        data_train, data_val, data_test = (x_train, y_train), (x_val, y_val), (x_test, y_test),
        save_metrics(model, data_train, data_val, data_test, _denormalizer, lb, ub, save_path, names)

        # 4. Save losses
        save_losses(history, save_path)

        # 5. Save plot
        data_total, data_train_val, data_test = (total_x, total_y), (train_x, train_y), (test_x, test_y)
        save_plot_obs_sim(model, data_total, data_train_val, data_test,
                          _normalizer, _denormalizer, lb, ub, rAvg, rStd, names, save_path)

        # 6. Save SHAP value plot
        x_train_shap = np.copy(train_x)
        x_train_shap = _normalizer(x_train_shap, rAvg, rStd)
        """if fold == 1:
            save_shap_values(model, x_train_shap, x_test, cluster, save_path)"""

        fold += 1


def train_hybrid_model(seq, nvars, ds, nout, act, x_train, y_train, x_val, y_val, epochs, bs, lr, ft, ks, h):
    # define model structure
    model = build_hybrid_model(nvars=nvars, nout=nout, seq=seq, ft=ft, ks=ks, ds=ds, act=act, h=h, mode='normal')

    # training
    optimizer = Adam(learning_rate=lr, epsilon=10E-3)
    model.compile(loss=custom_loss, optimizer=optimizer)  # custom_loss

    # early stopping
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                 verbose=0, patience=15, restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='min',
                                         min_delta=0.005, cooldown=0, min_lr=lr / 100)

    tnan = callbacks.TerminateOnNaN()

    # fit network
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, verbose=2, batch_size=bs, callbacks=[es, reduce, tnan])
    return model, history


def train_hybrid_model_kfold(seq, pt, ft, ks, ds, act, isTraining, toNorm, lr, epochs, bs, start='2011-01-01',
                             end='2021-04-07', cluster=None, save_base=None, h='linear', toResample=False, freq='1D'):
    # load data
    input_path = '../../Data/Calibration/inputs.xlsx'
    heads_path = '../../Data/Calibration/reference_head.xlsx'

    t, names, total_x, total_y, train_x, train_y, test_x, test_y = load_data(input_path,
                                                                             seq=seq, pt=pt,
                                                                             head_path=heads_path,
                                                                             isTraining=isTraining,
                                                                             cluster=cluster,
                                                                             toResample=toResample,
                                                                             freq=freq)

    # split into cross-validation datasets
    k = 10
    kFold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    for idx_train, idx_val in kFold.split(X=train_x, y=train_y):
        print('Start training fold %d/%d' % (fold, k))
        save_path = save_base + str(fold) + '/'

        # get training and validation data
        x_train, y_train = train_x[idx_train], train_y[idx_train]
        x_val, y_val = train_x[idx_val], train_y[idx_val]

        # inputs and outputs dimensions
        inp_dim = x_train.shape[-1]
        out_dim = y_train.shape[-1]

        # normalize (min, max) or standardize
        size = int(inp_dim / 2)

        # inputs characteristics
        rReshape = np.reshape(x_train, (-1, x_train.shape[-1]))

        if toNorm:
            lb, ub = np.mean(y_train, axis=0), np.std(y_train, axis=0)
            rAvg, rStd = np.mean(rReshape, axis=0), np.std(rReshape, axis=0)
        else:
            lb, ub = np.min(y_train, axis=0), np.max(y_train, axis=0)
            rAvg, rStd = np.min(rReshape, axis=0), np.max(rReshape, axis=0)

        _normalizer, _denormalizer = normalizer(toNorm=toNorm), denormalizer(toNorm=toNorm)

        # normalize only y
        y_train = _normalizer(y_train, lb, ub)
        y_val = _normalizer(y_val, lb, ub)
        y_test = _normalizer(test_y, lb, ub)

        # x_test
        x_test = np.copy(test_x)

        # build the model architecture
        nvars = inp_dim
        nout = out_dim

        # train model
        model, history = train_hybrid_model(seq, nvars, ds, nout, act, x_train, y_train, x_val, y_val, epochs, bs, lr,
                                            ft, ks, h)

        # save data
        # 1. save the model
        save_model_path = save_path + 'model'
        model.save_weights(filepath=save_model_path)

        # 2. Hyper-params
        hyper_params_dict = {
            'seq': seq, 'nvars': nvars, 'ds': ds, 'nout': nout, 'act': act, 'lb': lb, 'ub': ub, 'ft': ft, 'ks': ks,
            'toNorm': toNorm, 'h': h, 'names': names.values}

        save_hyper_params(hyper_params_dict, save_path)

        # 3. Save metrics
        data_train, data_val, data_test = (x_train, y_train), (x_val, y_val), (x_test, y_test),
        save_metrics(model, data_train, data_val, data_test, _denormalizer, lb, ub, save_path, names)

        # 4. Save losses
        save_losses(history, save_path)

        # 5. Save plot
        data_total, data_train_val, data_test = (total_x, total_y), (train_x, train_y), (test_x, test_y)
        save_plot_obs_sim(model, data_total, data_train_val, data_test,
                          _normalizer, _denormalizer, lb, ub, rAvg, rStd, names, save_path, isCNN=False)

        # 6. Save SHAP value plot
        x_train_shap = np.copy(train_x)
        save_shap_values(model, x_train_shap, x_test, cluster, save_path)

        fold += 1
