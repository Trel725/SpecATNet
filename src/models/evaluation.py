import pickle
import pandas as pd
import torch
from tqdm.auto import tqdm
import numpy as np
from data_utils import train_test_split_samples
from common import device
import mlflow

def accuracy(pred, true):
    return ((pred > 0.5) == true).float().mean()

def predict_by_group(X, samples, model, slen=8, repetitions=5):
    X_ = X.copy()
    X_["sample"] = samples
    X_gr = X_.groupby("sample")
    predictions = {}
    for sample, spectra in X_gr:
        spectra = spectra.drop(columns="sample")
        if spectra.shape[0] < slen:
            spectra = spectra.sample(n=slen, replace=True)
        res = []
        for _ in range(repetitions):
            spectra = spectra.sample(frac=1.0)
            batch = _create_batch(spectra.values, slen)
            preds = model(torch.Tensor(batch[:, :, None]).to(device))
            res.append(preds.detach().cpu().sigmoid().numpy())
        predictions[sample] = np.vstack(res)
    return predictions


def evaluate_preds(Y, samples, preds):
    Y_ = Y.copy()
    Y_['sample'] = samples
    labels = Y_.groupby('sample').agg(lambda x: x.iloc[0])
    pred_results = labels.apply(
        lambda x: x.values[None, :] == (preds[x.name] > 0.5), axis=1)
    return pred_results.apply(lambda x: x.mean())


def normalize_preds(Y, samples, preds):
    Y_ = Y.copy()
    Y_['sample'] = samples
    labels = Y_.groupby('sample').agg(lambda x: x.iloc[0])
    y_pred_dfs = []
    y_true_dfs = []
    for key in preds:
        y_pred = pd.DataFrame(preds[key], columns=Y.columns)
        y_pred['sample'] = key
        y_true = pd.DataFrame(np.repeat(labels.loc[key].values[None,:], y_pred.shape[0], axis=0), columns=Y.columns)
        y_true['sample'] = key
        y_pred_dfs.append(y_pred)
        y_true_dfs.append(y_true)
    return pd.concat(y_true_dfs), pd.concat(y_pred_dfs)


def summarize_per_seqlens(X, Y, samples, models, seqlens, random_state=42):
    summary_per_slen = {}
    for slen in tqdm(seqlens):
        true_summary = []
        pred_summary = []
        for (X_train, X_test,
             Y_train, Y_test,
             samples_train, samples_test,
             split_idx) in train_test_split_samples(X, Y, samples, random_state=random_state):
            preds = predict_by_group(X_test, samples_test, models[split_idx], slen=slen)
            y_true, y_pred = normalize_preds(Y_test, samples_test, preds)
            true_summary.append(y_true)
            pred_summary.append(y_pred)
        summary_per_slen[slen] = dict(true=pd.concat(true_summary).sort_values(by='sample'),
                                      pred=pd.concat(pred_summary).sort_values(by='sample'))
    with open("summary_per_slen.pickle", "wb") as f:
        pickle.dump(summary_per_slen, f)

    mlflow.log_artifact("summary_per_slen.pickle", "eval")



def _create_batch(arr, slen):
    num_splits = np.floor(arr.shape[0] / slen)
    return np.array(np.split(arr[:int(num_splits*slen)], num_splits))