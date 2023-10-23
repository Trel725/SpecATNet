import pandas as pd
import numpy as np
import sklearn
import pickle
import os
from scipy.signal import savgol_filter


def subt_back(s, lam=1e6):
    x = np.arange(s.size)
    subtracted, base = rampy.baseline(x, y, np.array(
        [[0., x[-1]]]), method="als", lam=smoothness)
    return subtracted


def densenet_block(inp, aux_inps, filters=8, kern_size=(10,), strides=(1,)):
    '''
    use it like:

    out, aux_out = densenet_block(inp, [], strides=4)
    aux_outs = [aux_out]
    outs = [out]

    for ksize in [(8,), (8,)]:
        out, aux_out = densenet_block(out, aux_outs, kern_size=ksize)
        aux_outs.append(aux_out)
    '''
    x = Conv1D(filters, kern_size, strides=strides, padding="same",
               kernel_regularizer=ortho_regularizer)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    if len(aux_inps) > 0:
        return Concatenate()([x] + aux_inps), x
    else:
        return x, x


def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


'''
def norm_func(x, a=0, b=1):
    return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a


def normalize(x, y):
    x = np.apply_along_axis(norm_func, axis=1, arr=x)
    mask = ~np.isnan(x).any(axis=1)
    return x[mask], y[mask]
'''


def csvs_to_dataframe(files):
    dfs = []
    keys = []
    idx_to_labels = {}
    for idx, file in enumerate(files):
        data = pd.read_csv(file)
        dfs.append(data)
        key = np.ones(data.shape[0]) * idx
        keys.append(key)
        idx_to_labels[idx] = os.path.abspath(file)

    df = pd.concat(dfs)
    keys = np.concatenate(keys)
    df['key'] = keys
    return df, idx_to_labels


def visualize_high_dim(datax, keys, interactive=False, pca_comp=30):
    tsne = TSNE(n_jobs=16)
    pca = sklearn.decomposition.PCA(n_components=pca_comp)

    compressed = pca.fit_transform(datax)
    print("{} variance explained".format(pca.explained_variance_ratio_.sum()))

    embedded = tsne.fit(compressed)
    if interactive:
        scatter_with_viewer(embedded[:, 0], embedded[:, 1],
                            spectra=datax, c=keys, n_colors=np.unique(keys).shape[0], s=3)
    else:
        plt.scatter(embedded[:, 0], embedded[:, 1], c=keys, s=3)
    plt.colorbar()
    plt.show()


def spickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle(file):
    with open(file, "rb") as f:
        res = pickle.load(f)
    return res


def bokeh_scatter(figure, x, y, c):
    # figure: bokeh figure
    # x, y, c - coordinates, color label
    if c.dtype.type is not np.string_:
        c = [str(i) for i in c]
        c = np.array(c)
    palette = bokeh.palettes.d3['Category10'][len(np.unique(c))]
    # color_map = bokeh.models.CategoricalColorMapper(factors=np.unique(c),
    #                                                palette=palette)
    figure.scatter(x, y, color=palette[c])


def reinit_cuda():
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


def style1(size=(5, 3)):
    import matplotlib
    plt.style.use("ggplot")

    params = {
        'axes.labelpad': 1.0,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': size
    }

    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    matplotlib.rcParams.update(params)


def style1_smaller(size=(5, 3)):
    import matplotlib
    plt.style.use("ggplot")

    params = {
        'axes.labelpad': 1.0,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': False,
        'figure.figsize': size
    }

    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    matplotlib.rcParams.update(params)


def style1_small(size=(5, 3)):
    import matplotlib
    plt.style.use("ggplot")

    params = {
        'axes.titlesize': 7,
        'axes.labelpad': 0.5,
        'axes.labelsize': 5,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'text.usetex': False,
        'figure.figsize': size
    }

    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    matplotlib.rcParams.update(params)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def mysignaltonoise(a, length=11, order=3):
    a = norm_func(a)
    filtered = savgol_filter(a, length, order)
    return np.abs((a - filtered)).sum()


def autocorr(spec):
    ln = spec.shape[0]
    return np.fft.ifft(np.abs(np.fft.fft(spec)**2))[:ln // 2]


def norm_func(x, a=0, b=1):
    return ((b - a) * (x - min(x))) / (max(x) - min(x) + 1e-7) + a


def normalize(x, y=None):
    x = np.apply_along_axis(norm_func, axis=1, arr=x)
    #x = np.apply_along_axis(autocorr, axis=1, arr=x)
    mask = ~np.isnan(x).any(axis=1)
    if y is not None:
        return x[mask], y[mask]
    else:
        return x[mask]


def cuda_memgrowth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
