import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_data(input_path, seq, pt=None, isTraining=False, head_path=None,
              start='2011-01-01', end='2021-04-07', cluster=None,
              toResample=False, freq='3D'):
    # load climatic variable
    prcp_ = pd.read_excel(input_path, 0, index_col=0)
    evap_ = pd.read_excel(input_path, 1, index_col=0)

    prcp_ = prcp_[(prcp_['t'] >= start) & (prcp_['t'] <= end)]
    evap_ = evap_[(evap_['t'] >= start) & (evap_['t'] <= end)]

    # fill nan
    prcp_.fillna(method='ffill', inplace=True)
    evap_.fillna(method='ffill', inplace=True)

    # reindex
    prcp_.set_index(prcp_['t'], inplace=True)
    evap_.set_index(evap_['t'], inplace=True)

    # resample
    if toResample:
        prcp_ = prcp_.resample(freq).mean()
        evap_ = evap_.resample(freq).mean()

        prcp_['t'] = prcp_.index
        evap_['t'] = evap_.index

    # get time and delete time column
    t = prcp_['t']
    prcp_.pop('t')
    evap_.pop('t')

    # use only a given cluster of station (if necessary)
    if cluster is not None:
        prcp_ = prcp_[cluster]
        evap_ = evap_[cluster]

    names = prcp_.columns

    # convert into numpy array
    prcp = np.array(prcp_).reshape((-1, len(cluster)))
    evap = np.array(evap_).reshape((-1, len(cluster)))

    # concatenate inputs
    atmo = np.concatenate([prcp, evap], axis=-1)

    # format data
    window_width = atmo.shape[1]
    total_x = np.lib.stride_tricks.sliding_window_view(atmo, window_shape=(seq, window_width)).squeeze()

    if not (t.iloc[seq:].values.shape[0] == total_x.shape[0]):
        total_x = total_x[:-1, :, :]

    # train and test dataset
    end_train = int(pt * total_x.shape[0])
    train_x = total_x[:end_train, :, :]
    test_x = total_x[end_train:, :, :]

    # load head if necessary
    total_y, train_y, test_y = [], [], []
    if isTraining:
        assert head_path is not None
        head = pd.read_excel(head_path, 0, index_col=0)
        head.fillna(method='ffill', inplace=True)
        head = head[(head['t'] >= start) & (head['t'] <= end)]

        # reindex
        head.set_index(head['t'], inplace=True)

        # resample
        if toResample:
            head = head.resample(freq).mean()
            head['t'] = head.index

        head.pop('t')

        if cluster is not None:
            head = head[cluster]

        head = np.array(head)

        size = head.shape[0]
        for j in range(size):
            end_idx = j + seq
            if end_idx >= size:
                break

            seq_y = head[end_idx, :]
            total_y.append(seq_y)

        total_y = np.array(total_y)

        train_y = total_y[:end_train, :]
        test_y = total_y[end_train:, :]

    # final sequence of t
    t = t.iloc[seq:].values

    # release memory
    del [prcp_]
    del [evap_]

    del [prcp]
    del [evap]
    del [atmo]

    # return
    if isTraining:
        return t, names, total_x, total_y, train_x, train_y, test_x, test_y
    else:
        return t, names, total_x


# same as previous but only the way the Excel files are loaded is modified
# also only total_x is retrieved
def load_data_projection(input_path, seq, start='1990-01-01', end='2081-12-31', cluster=None,
                         toResample=False, freq='3D'):
    # load climatic variable: by this way, the data is loaded only one time instead of two time as previously
    xls_data = pd.ExcelFile(input_path)

    prcp_ = pd.read_excel(xls_data, 0, index_col=0)
    evap_ = pd.read_excel(xls_data, 1, index_col=0)

    prcp_ = prcp_[(prcp_['t'] >= start) & (prcp_['t'] <= end)]
    evap_ = evap_[(evap_['t'] >= start) & (evap_['t'] <= end)]

    # fill nan
    prcp_.fillna(method='ffill', inplace=True)
    evap_.fillna(method='ffill', inplace=True)

    # reindex
    prcp_.set_index(prcp_['t'], inplace=True)
    evap_.set_index(evap_['t'], inplace=True)

    # resample
    if toResample:
        prcp_ = prcp_.resample(freq).mean()
        evap_ = evap_.resample(freq).mean()

        prcp_['t'] = prcp_.index
        evap_['t'] = evap_.index

    # get time and delete time column
    t = prcp_['t']
    prcp_.pop('t')
    evap_.pop('t')

    # use only a given cluster of station (if necessary)
    if cluster is not None:
        prcp_ = prcp_[cluster]
        evap_ = evap_[cluster]

    names = prcp_.columns

    # convert into numpy array
    prcp = np.array(prcp_).reshape((-1, len(cluster)))
    evap = np.array(evap_).reshape((-1, len(cluster)))

    # concatenate inputs
    atmo = np.concatenate([prcp, evap], axis=-1)

    # format data
    window_width = atmo.shape[1]
    total_x = np.lib.stride_tricks.sliding_window_view(atmo, window_shape=(seq, window_width)).squeeze()

    if not (t.iloc[seq:].values.shape[0] == total_x.shape[0]):
        total_x = total_x[:-1, :, :]

    # final sequence of t
    t = t.iloc[seq:].values

    # release memory
    del [prcp_]
    del [evap_]

    del [prcp]
    del [evap]
    del [atmo]

    # return
    return t, names, total_x
