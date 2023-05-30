from basics_functions import get_cluster, delete_files_in_dict
from train_loop import train_trad_model_kfold, train_hybrid_model_kfold


def delete_files(base_path):
    for M in range(1, 11):
        delete_files_in_dict(base_path + str(M))


def execute_trad(aq='uc', seq=80, pt=0.8, ft=2, ks=11, ds=3, act='tanh', isTraining=True, toNorm=True,
                 lr=5e-3, epochs=100, bs=126, toResample=False, freq='1D', ct=1):
    save_base = '../../Results/Training/Temporary/Traditional/' + aq.upper() + '/M'
    cluster = get_cluster(aq, ct)
    delete_files(save_base)

    train_trad_model_kfold(seq, pt, ft, ks, ds, act, isTraining, toNorm, lr, epochs, bs, start='2011-01-01',
                           end='2021-04-07', cluster=cluster, save_base=save_base, toResample=toResample, freq=freq)


def execute_hybrid(aq='uc', seq=80, pt=0.8, ft=2, ks=11, ds=3, act='tanh', isTraining=True, toNorm=True,
                   lr=5e-3, epochs=100, bs=126, h='linear', toResample=False, freq='1D', ct=1):
    save_base = '../../Results/Training/Temporary/Hybrid/' + aq.upper() + '/M'
    cluster = get_cluster(aq, ct)
    delete_files(save_base)

    train_hybrid_model_kfold(seq, pt, ft, ks, ds, act, isTraining, toNorm, lr, epochs, bs, start='2011-01-01',
                             end='2021-04-07', cluster=cluster, save_base=save_base, h=h, toResample=toResample,
                             freq=freq)


# execute_trad(aq='uc', ft=17, ks=11, ds=17, act='tanh', lr=5e-3, epochs=50, bs=126, toResample=False, freq='1D', ct=1)

def run_trad(aq='uc', ft=1, ks=11, ds=3, act='tanh', epochs=80, bs=126, ct=3, seq=80, pt=0.8):
    execute_trad(aq=aq, ft=ft, ks=ks, ds=ds, act=act, lr=5e-3, epochs=epochs, bs=bs, toResample=False, freq='1D',
                 ct=ct, seq=seq, pt=pt)


def run_hybrid(aq='uc', ft=1, ks=11, ds=3, act='tanh', epochs=80, bs=126, ct=3, seq=80, h='linear', pt=0.8):
    execute_hybrid(aq=aq, ft=ft, ks=ks, ds=ds, act=act, lr=5e-3, epochs=epochs, bs=bs, h=h, toResample=False,
                   freq='5D', ct=ct, seq=seq, pt=pt)


run_trad(aq='c', ft=5, ks=11, ds=3, act='tanh', epochs=60, bs=128, ct=2, seq=80)

"""run_hybrid(aq='c', pt=0.8, ft=5, ks=11, ds=3, act='tanh', epochs=60, bs=128,
           ct=2, seq=80, h='complete_nc')  # bs= epoch=60 (article) complete_nc"""
