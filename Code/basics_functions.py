import tensorflow as tf
import pandas as pd
import pickle as pkl
import shap
import hydroeval as he
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pylab import *
from model_builder import build_hybrid_model, build_trad_model


def delete_files_in_dict(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def custom_loss(y_true, y_pred):
    numerator = tf.reduce_sum(tf.square(y_true - y_pred), axis=None)
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=None)))
    rNSE = tf.math.divide_no_nan(numerator, denominator)
    loss = rNSE

    return loss


def normalizer(toNorm=True, a=2.0, b=1.0):
    if toNorm:
        def normalize(y, yAvg, yStd):
            return (y - yAvg) / yStd
    else:
        def normalize(y, y_min, y_max):
            return a * (y - y_min) / (y_max - y_min) - b

    return normalize


def denormalizer(toNorm=True, a=2.0, b=1.0):
    if toNorm:
        def denormalize(y, yAvg, yStd):
            return yAvg + yStd * y
    else:
        def denormalize(y, y_min, y_max):
            return (b + y) * (y_max - y_min) / a + y_min

    return denormalize


def get_cluster(aquifer='uc', ct=1):
    assert aquifer == 'uc' or aquifer == 'sc' or aquifer == 'c'
    path_data = '../../Data/Calibration/clusters_%s.xlsx' % (aquifer,)

    # select pz of 'aquifer' of 'cluster'
    ct_data = pd.read_excel(path_data, index_col=0)
    ct_data['id'] = ct_data['id'].astype('str')
    ct_data['id'] = '0' + ct_data['id']

    # rename columns
    # ct_data.columns = ['id', 'x', 'y', 'aq', 'ct']
    ct_data.columns = ['id', 'x', 'y', 'ct']

    # ct_data = ct_data[ct_data['aq'] == aq[aquifer]]
    if aquifer != 'sc':
        ct_data = ct_data[ct_data['ct'] == ct]

    cluster = ct_data['id']

    if cluster.empty:
        raise Exception('Cluster %d is not a valid cluster number' % (ct,))
    else:
        return cluster.values


def get_metrics(y_true, y_pred, names):
    # 1. NSE
    # 2. Kling-Gupta KGE
    # 3. R (Pearson correlation)
    # 4. RMSE
    # 5. PBIAS
    nse, kge, r, nrmse, pbias = [], [], [], [], []

    size = y_true.shape[1]
    for j in range(size):
        sim = y_pred[:, j]
        obs = y_true[:, j]

        _nse = he.evaluator(he.nse, sim, obs)
        _kge, _r, _, _ = he.evaluator(he.kge, sim, obs)
        _rmse = he.evaluator(he.rmse, sim, obs)
        _pbias = he.evaluator(he.pbias, sim, obs)

        # min and max of obs
        min_obs = np.min(obs, axis=0)
        max_obs = np.max(obs, axis=0)

        # compute nrmse
        _nrmse = 100 * np.nan_to_num(_rmse / (max_obs - min_obs))

        nse.append(_nse[0])
        kge.append(_kge[0])
        r.append(_r[0])
        nrmse.append(_nrmse[0])
        pbias.append(_pbias[0])

    metrics = pd.DataFrame(np.zeros((size, 6)), columns=['id', 'nse', 'kge', 'r', 'rmse', 'pbias'])
    metrics['id'] = names
    metrics['nse'] = nse
    metrics['kge'] = kge
    metrics['r'] = r
    metrics['nrmse'] = nrmse
    metrics['pbias'] = pbias
    return metrics


def save_metrics(model, data_train, data_val, data_test,
                 _denormalizer, lb, ub, save_path, names):
    # input and output data are already normalized for CNN and not for TgNN (not necessary)
    x_train, y_train = data_train
    x_val, y_val = data_val
    x_test, y_test = data_test

    # predict
    yp_train = predict(x_train, model, True, _denormalizer, lb, ub)
    yp_val = predict(x_val, model, True, _denormalizer, lb, ub)
    yp_test = predict(x_test, model, True, _denormalizer, lb, ub)

    # denormalize y_obs
    yo_train = _denormalizer(y_train, lb, ub)
    yo_val = _denormalizer(y_val, lb, ub)
    yo_test = _denormalizer(y_test, lb, ub)

    # get metrics
    metrics_train = get_metrics(yo_train, yp_train, names)
    metrics_val = get_metrics(yo_val, yp_val, names)
    metrics_test = get_metrics(yo_test, yp_test, names)

    # save metrics
    metrics_train.to_excel(save_path + 'metrics_train.xlsx')
    metrics_val.to_excel(save_path + 'metrics_val.xlsx')
    metrics_test.to_excel(save_path + 'metrics_test.xlsx')

    # box plot
    fs = 10
    plt.rcParams.update({'font.size': fs, 'font.family': 'Arial'})
    fig, ax = plt.subplots()

    colors_for_boxplot = {
        'black': '#000000',
        'blue': '#3275a1',
        'orange': '#e1802c',
        'green': '#3a923a',
        'red': '#c03d3d',
        'purple': '#9372b2',
        'gray': '#c4c4c4'
    }
    boxes = ax.boxplot(x=metrics_test['nse'].values)

    # Customize border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_edgecolor(colors_for_boxplot['gray'])
    ax.spines['left'].set_edgecolor(colors_for_boxplot['gray'])

    # get kge and nrmse
    kge = np.round(metrics_test['kge'].values[0], 2)
    nrmse = np.round(metrics_test['nrmse'].values[0], 2)

    for line in boxes['medians']:
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        text(x + 0.02, y, '%.2f' % y, fontsize=fs, verticalalignment='center')

    plt.xticks([1], ['Test'])
    plt.ylabel('NSE [-]')
    # plt.ylim([-1, 1])
    plt.title('KGE: ' + str(kge) + ', NRMSE: ' + str(nrmse) + '%')
    plt.tight_layout()
    plt.savefig(save_path + 'nse.png', dpi=350)
    plt.clf()


def save_hyper_params(hyper_params, save_path):
    with open(save_path + 'hyper_params.txt', 'wb') as gFile:
        pkl.dump(hyper_params, gFile)


def save_losses(history, save_path):
    loss = np.reshape(np.array(history.history['loss']), (-1, 1))
    val_loss = np.reshape(np.array(history.history['val_loss']), (-1, 1))

    try:
        loss = np.concatenate([loss, val_loss], axis=-1)
        loss = pd.DataFrame(loss, columns=['Training', 'Testing'])

        path = save_path + 'losses.xls'
        loss.to_excel(path)
    except:
        print('...')


def save_plot_obs_sim(model, data_total, data_train_val, data_test,
                      _normalizer, _denormalizer, lb, ub, rAvg, rStd, names, save_path, isCNN=True):
    # input and output data are not normalized
    # 1. get data
    _, y_total = data_total
    x_train, y_train = data_train_val
    x_test, y_test = data_test

    # 2. Normalize input only for CNN
    if isCNN:
        x_train = _normalizer(x_train, rAvg, rStd)
        x_test = _normalizer(x_test, rAvg, rStd)

    # 3. Predict
    yp_train = predict(x_train, model, True, _denormalizer, lb, ub)
    yp_test = predict(x_test, model, True, _denormalizer, lb, ub)

    # 4. time axis
    t_total = list(range(len(y_total)))
    t_train = list(range(len(x_train)))
    t_test = list(range(len(x_train), len(x_train) + len(x_test)))

    out_dim = y_test.shape[-1]

    n = 1
    k = 1
    color = ['#000000', '#f58e20', None]
    save_name = 'OS_' + str(k) + '.png'
    a, b = 3, 3

    plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})
    plt.figure(figsize=(20, 10))
    for j in range(out_dim):
        name = names[j]
        plt.subplot(a, b, n)

        # total
        plt.plot(t_total, y_total[:, j], color=color[0], linewidth=1.5)

        # train
        plt.plot(t_train, yp_train[:, j], color=color[1], linewidth=1.5)

        # test
        plt.plot(t_test, yp_test[:, j], color=color[2], linewidth=1.5)

        plt.xlabel('Time [days]')
        plt.ylabel('GWL [m]')
        plt.title(name)

        if (n == a * b) or (j == out_dim - 1):
            n = 0
            k += 1
            plt.tight_layout()
            plt.savefig(save_path + save_name, dpi=300)
            save_name = 'OS_' + str(k) + '.png'
            plt.clf()

        n = n + 1


def save_shap_values(model, x_train, x_test, name, save_path):
    # background data
    n = 1
    P = int(100 * n)  # 4
    background = x_train[:P, :, :]

    # test data
    N = int(100 * n)  # 500
    x_shap = x_test[:N, :, :]
    x_shp_reshaped = x_shap.reshape(-1, x_shap.shape[-1])
    n_size = int(x_shap.shape[-1] / 2)

    # shape value
    explainer = shap.DeepExplainer(model=model, data=background)
    shap_value = explainer.shap_values(X=x_shap, check_additivity=False)
    print('Shap length: ' + str(len(shap_value)))

    labels = []
    no_feats = int(x_test.shape[-1] / 2)
    for i in range(2 * no_feats):
        if i < no_feats:
            if i + 1 < 10:
                labels.append('vi 0' + str(i + 1))
            else:
                labels.append('vi ' + str(i + 1))
        else:
            if (i - no_feats + 1) < 10:
                labels.append('ep 0' + str(i - no_feats + 1))
            else:
                labels.append('ep ' + str(i - no_feats + 1))

    for j in range(n_size):
        shap_reshape = shap_value[j].reshape(-1, shap_value[j].shape[-1])

        # plot
        plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
        # shap.summary_plot(shap_reshape, x_shp_reshaped, feature_names=['vi', 'ep'], show=False)
        shap.summary_plot(shap_reshape, x_shp_reshaped, feature_names=labels, show=False, max_display=8)
        plt.xlabel("SHAP value (impact on GWL)")
        plt.title(name[j])
        plt.tight_layout()
        plt.savefig(save_path + 'shap' + str(j) + '.png', dpi=300)
        plt.clf()


def predict(x, model, toDenorm, _denormalizer, lb, ub):
    y = model.predict(x)
    if toDenorm:
        y = _denormalizer(y, lb, ub)
    return y


def load_model_and_params_cnn(file_path):
    # load hyper-parameters
    fn = file_path + 'hyper_params.txt'
    with open(fn, 'rb') as g:
        params = pkl.load(g)

    seq, lb, ub = params['seq'], params['lb'], params['ub']
    rAvg, rStd = params['rAvg'], params['rStd']
    toNorm, ds, ft, ks = params['toNorm'], params['ds'], params['ft'], params['ks']
    nvars, act, nout = params['nvars'], params['act'], params['nout']
    names = params['names']

    # load model
    save_model_path = file_path + 'model'
    model = build_trad_model(seq, nvars, ft, ks, ds, nout, act)
    model.load_weights(save_model_path)
    return model, seq, nvars, ft, ks, ds, nout, act, lb, ub, rAvg, rStd, toNorm, names


def load_model_and_params_TgNN(file_path, mode='normal', inter_vars='qg'):
    # load hyper-parameters
    fn = file_path + 'hyper_params.txt'
    with open(fn, 'rb') as g:
        params = pkl.load(g)

    seq, lb, ub, ds = params['seq'], params['lb'], params['ub'], params['ds']
    nvars, act, nout = params['nvars'], params['act'], params['nout']
    ft, ks, h, toNorm = params['ft'], params['ks'], params['h'], params['toNorm']
    names = params['names']

    # load model
    save_model_path = file_path + 'model'

    # define model structure
    model = build_hybrid_model(nvars=nvars, nout=nout, seq=seq, ft=ft, ks=ks, ds=ds, act=act, h=h, mode=mode,
                               inter_vars=inter_vars)

    model.load_weights(save_model_path)
    return model, seq, nvars, ds, nout, act, lb, ub, toNorm, names
