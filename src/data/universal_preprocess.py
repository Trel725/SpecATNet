#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sklearn.decomposition
from scipy.ndimage import binary_dilation
import os
import argparse
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import savgol_filter
import custom_utils as utils
from pathlib import Path

here = str(Path(__file__).resolve().parent)


def resample(df: pd.DataFrame):
    xold = df.loc['x_wvl']
    newdf = pd.DataFrame(df.drop("x_wvl").apply(
        lambda y: np.interp(wvl, xold, y), axis=1).tolist())
    return newdf


def csvs_to_dataframe(files):
    dfs = []
    keys = []
    idx_to_labels = {}
    for idx, file in enumerate(files):
        data = pd.read_csv(file)
        if "Index" in data:
            data.set_index("Index", inplace=True)
        if args.resample:
            if "x_wvl" not in data.index:
                continue
            data = resample(data)
        dfs.append(data)
        key = np.ones(data.shape[0]) * idx
        keys.append(key)
        idx_to_labels[idx] = os.path.abspath(file)

    df = pd.concat(dfs)
    keys = np.concatenate(keys)
    df['key'] = keys
    return df, idx_to_labels


def grouped_csvs_to_dataframe(grp_files, names):
    dfs = []
    idx_to_labels = {}
    if len(names) != len(grp_files):
        print("Group names are not provided, using file names instead...")
        names = None
    for idx, files in enumerate(grp_files):
        df, labels_dict = csvs_to_dataframe(files)
        if args.resample:
            df = resample(df)
        df['key'] = idx
        dfs.append(df)
        if names is not None:
            idx_to_labels[idx] = names[idx]
        else:
            idx_to_labels[idx] = " ".join(
                [os.path.basename(v) for v in labels_dict.values()])
    return pd.concat(dfs), idx_to_labels


parser = argparse.ArgumentParser(
    description='PreProcess spectral dataset with optional visualization')
parser.add_argument("-d", "--data", type=str, nargs='+', action='append',
                    required=True,
                    help='Define file groups, i.e. -d 1.csv 2.csv -d 3.csv 4.csv')

parser.add_argument("--names", type=str, nargs='+', default=[],
                    help='Define group names, i.e. --names A B')

parser.add_argument('-s', '--savepath', type=str, default="./",
                    help="Save the preprocessed data to savepath")

parser.add_argument('-o', "--outliers", action='store_true',
                    help="Try to detect outliers in the data")

parser.add_argument("--int-thresh", type=float,
                    help="Intensity threshold to separate data (instead of interactive)")

parser.add_argument("-m", "--manual", action='store_true',
                    help="Manually select data (interactive mode)")

parser.add_argument("-c", "--categorical", action='store_true',
                    help="Interpret key values as categories")

parser.add_argument("--dropnan", action="store_true",
                    help="Drop rows containing NaN and too large values")

parser.add_argument("-N", "--normalize", action='store_true',
                    help="Normalize data to 0..1 range")

parser.add_argument("--single-file", action='store_true',
                    help="Preprocess single file (w/o key column)")

parser.add_argument("--mapping-dict", type=str,
                    help="When reading single file, map keys from here")

parser.add_argument("-t", "--spikes-thresh", type=float, default=8.0,
                    help="Use another spike threshold (default is 8.0)")

parser.add_argument("--no-spikes-removal", action='store_true',
                    help="Skip spikes removal procedure")

parser.add_argument("--filter-metrics", type=str, default=None,
                    help="Override metric to filter spectra, e.g. np.sum")

parser.add_argument("--save-server", type=str, default=None, help="Save server object to file \
    for further hosting")

parser.add_argument(
    "--embedder", choices=['tsne', 'umap'], default="umap", help='Embedding algorithm to use')

parser.add_argument("--full-path", action="store_true",
                    help="Show full paths instead of basenames in plot")

parser.add_argument("--host", type=str, default=None,
                    help="Server object to host")

parser.add_argument("--origin", nargs="+",
                    default=["*"], help="Websocket origin to use")

parser.add_argument("--resample", action="store_true", help="Resample data (useful if different \
                    files have different wavelengths)")

parser.add_argument("--wvl", nargs=3, default=[100, 4278, 2090],
                    help="args to np.linspace defining common wavelengths")

parser.add_argument("--window", type=float, nargs='+', default=None, help="Window over which to calculate metrics \
                                                                           (relative, e.g. 0.5 1.0)")


def norm_func(x, a=0, b=1):
    return ((b - a) * (x - min(x))) / (max(x) - min(x) + 1e-7) + a


def normalize(x, y):
    x = np.apply_along_axis(norm_func, axis=1, arr=x)
    mask = ~np.isnan(x).any(axis=1)
    return x[mask], y[mask]


def autocorr(arr):
    l = arr.shape[0]
    if l == 0 or arr.var() == 0:
        return np.full_like(arr, np.nan, dtype=np.double)
    return np.fft.ifft(np.abs(np.fft.fft(arr - arr.mean()))**2).real / (arr.var() * l)


def noise_sg(arr):
    l = arr.shape[0]
    au = autocorr(arr)
    au = au[0:l // 2]
    if np.any(np.isnan(au)):
        return np.nan
    smooth = savgol_filter(au, 15, 3)
    return np.sum((au - smooth)**2)


def filter_by_autocorr(data, keys):

    noises = np.apply_along_axis(noise_sg, axis=1, arr=data)
    mask = ~np.isnan(noises)
    noises = noises[mask]
    data = data[mask, :]
    keys = keys[mask]
    thresh = float(input("Enter the autocorrelation threshold: \n"))

    good_mask = (noises < thresh)
    data = data[good_mask, :]
    keys = keys[good_mask]
    return data, keys


def despike(data, threshold=None, dilation_iter=1):
    threshold = args.spikes_thresh
    grad = np.gradient(data)
    med = np.median(grad)
    MAD = np.median(np.abs(grad - med))
    Z_score = 0.6745 * (grad - med) / (MAD + 1e-7)
    spikes = (np.abs(Z_score) > threshold)
    spikes = binary_dilation(spikes, iterations=dilation_iter)
    clean = data.copy()
    clean[spikes] = np.interp(
        np.where(spikes)[0], np.where(~spikes)[0], data[~spikes])
    return clean


def despike_arr(spectra, keys):
    for k in np.unique(keys):
        key_mask = (keys == k)
        spectra[key_mask, :] = np.apply_along_axis(
            despike, axis=1, arr=spectra[key_mask, :])
    return spectra


def resample_arr(spectra, keys):
    spectra = np.apply_along_axis(despike, axis=1, arr=spectra[key_mask, :])
    return spectra


def preprocess(spectra, keys):
    # spectra, keys = filter_by_autocorr(spectra, keys)
    intensities = spectra.sum(axis=1)

    thresh = args.int_thresh
    good_mask = (intensities > thresh)

    spectra = spectra[good_mask, :]
    keys = keys[good_mask]

    for k in np.unique(keys):
        key_mask = (keys == k)
        if not args.no_spikes_removal:
            spectra[key_mask, :] = np.apply_along_axis(
                despike, axis=1, arr=spectra[key_mask, :])
    return spectra, keys


if __name__ == "__main__":
    args = parser.parse_args()
    wvl = np.linspace(*[int(i) for i in args.wvl])

    if not (args.savepath or args.visualize or args.manual):
        print("Nothing to do!")
        exit()

    labels = None

    if len(args.data) == 1 and len(args.data[0]) == 1 and not args.single_file:
        data = pd.read_csv(args.data[0][0])
        if "Index" in data:
            data.set_index("Index", inplace=True)
        if args.mapping_dict:
            mapping = utils.unpickle(args.mapping_dict)
            labels = [os.path.splitext(os.path.basename(v))[0]
                      for v in mapping.values()]

    elif len(args.data) == 1 and len(args.data[0]) == 1 and args.single_file:
        data = pd.read_csv(args.data[0][0])
        if "Index" in data:
            data.set_index("Index", inplace=True)
        data['key'] = 0

    elif len(args.data) > 1 or len(args.data[0]) > 1:
        if len(args.data) == 1:
            data, mapping_dict = csvs_to_dataframe(args.data[0])
        elif len(args.data) > 1:
            data, mapping_dict = grouped_csvs_to_dataframe(
                args.data, args.names)
        utils.spickle(mapping_dict, args.savepath + "/mapping_dict.pkl")
        labels = [os.path.basename(v) for v in mapping_dict.values()]

    if "x_wvl" in data.index:
        print("Found X in data, using it...")
        wvl = data.loc["x_wvl"].values
        if len(wvl.shape) > 1:
            wvl = wvl[0]
        data.drop("x_wvl", inplace=True)
    if args.dropnan:
        thresh = 1e10
        print("Dropping {} NaN rows...".format(
            np.isnan(data).any(axis=1).sum()))
        data = data[~np.isnan(data).any(axis=1)]
        print("Dropping {} rows larger than {}".format(
            (np.abs(data) > thresh).any(axis=1).sum(), thresh))
        data = data[~(np.abs(data) > thresh).any(axis=1)]

    data = data.fillna(0.0)
    datax = data.values[:, :-1]
    key = data['key'].values

    if args.categorical:
        cat_names, encoded = np.unique(key, return_inverse=True)
        cat_names = np.array([str(i) for i in cat_names])
        key = encoded
    elif labels:
        cat_names = labels
    else:
        cat_names = None

    keys = key
    if not args.no_spikes_removal:
        datax_pr = despike_arr(datax, keys)
    else:
        datax_pr = datax

    datax_pr

    if args.savepath:
        df = pd.DataFrame(datax_pr)
        print("final data size:", df.shape)
        if not args.single_file:
            df['key'] = keys
        df.columns = df.columns.astype(str)
        df.to_parquet(args.savepath + os.path.sep + "preprocessed.parquet")

    if args.normalize:
        datax_pr, keys = normalize(datax_pr, keys)
