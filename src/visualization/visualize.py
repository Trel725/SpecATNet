import logging
import os
import pickle
from pathlib import Path

import click
import shutil
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.utils.class_weight import  compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)


THRESHOLD = 0.5

def get_last_run_id(exp_name):
    exps = mlflow.search_runs(
        experiment_names=[exp_name])
    last_run = exps.sort_values(by='end_time').iloc[-1]
    return last_run.run_id


def make_summary_metrics(data):
    '''generate 4 basic metrics (acc, prec, recall, f1)
    for all polymers'''
    scores = accuracy_score, precision_score, recall_score, f1_score
    score_names = ["accuracy", "precision", "recall", "f1"]
    metrics_summary = {}
    for score, name in zip(scores, score_names):
        metrics = {}
        for seqlen in data:
            samples = data[seqlen]['pred']['sample']
            # since there might be very different number of spectra
            # per every sample, we need to take this into account
            weights = compute_sample_weight("balanced", samples)
            pred = data[seqlen]['pred'].drop(columns="sample")
            true = data[seqlen]['true'].drop(columns="sample")
            metrics[seqlen] = [score(true[poly], pred[poly] > 0.5, sample_weight=weights)
                               for poly in true.columns]
        metr_df = pd.DataFrame(metrics).T
        metr_df.columns = true.columns
        metrics_summary[name] = metr_df
    pd.concat(metrics_summary, axis=1).to_excel("results/metrics_summary.xlsx")
    mlflow.log_artifacts
    for metric in metrics_summary:
        metrics_summary[metric].plot()
        plt.xlabel("Sequence length")
        plt.ylabel(metric)
        plt.savefig(f"results/plots/{metric}.png", dpi=200)
    return metrics_summary['f1'].loc[max([s for s in data])].mean()


def make_average_pred(data):
    """generate averaged predictions"""
    for seqlen, d in data.items():
        sample = d['pred']['sample']
        binarized = (d['pred'].drop(columns='sample') > THRESHOLD)
        av_preds = binarized.groupby(sample).agg("mean")
        av_preds.to_excel(f"results/av_preds/average_preds_{seqlen}.xlsx")


def soft_accuracy(true, pred):
    '''soft accuracy: any overlapping class
       hard accuracy: all classes must overlap
    '''
    return (true == pred).mean().mean()


def make_acc_per_sample(data, mapping):
    """calculates accuracy for every sample individually"""
    acc_summary = {}
    for seqlen in data:
        pred = data[seqlen]['pred']
        true = data[seqlen]['true']
        acc_per_sample = {}
        for (sample1, t), (sample2, p) in zip(true.groupby("sample"), pred.groupby("sample")):
            if sample1 != sample2:
                raise ValueError()
            acc_per_sample[sample1] = soft_accuracy(t.drop(columns='sample'),
                                                     p.drop(columns='sample') > THRESHOLD)

        acc_summary[seqlen] = pd.Series(acc_per_sample)
    final_acc = pd.DataFrame(acc_summary)
    final_acc['sample'] = pd.Series(final_acc.index, index=final_acc.index)\
        .apply(lambda x: mapping[x] if x in mapping else None)
    final_acc.to_excel("results/acc_per_sample.xlsx")
    return acc_summary[max([s for s in data])].mean()


def generate_curves(data, curve_function):
    metrics = {}
    for seqlen in data.keys():
        samples = data[seqlen]['pred']['sample']
        weights = compute_sample_weight("balanced", samples)
        pred = data[seqlen]['pred'].drop(columns="sample")
        true = data[seqlen]['true'].drop(columns="sample")
        newx = np.linspace(0, 1, 512)
        df = pd.DataFrame(newx, columns=['x'])
        for poly in true.columns:
            x, y = curve_function(true[poly], pred[poly], sample_weight=weights)[:2]
            newy = np.interp(newx, x, y)
            df[poly] = newy
        metrics[seqlen] = df
    return metrics


def make_curves(data):
    for curve_fcn, name in zip([roc_curve, precision_recall_curve], ['roc', "precrec"]):
        for seqlen, curves in generate_curves(data, curve_fcn).items():
            curves.plot(x='x')
            plt.title(name)
            plt.savefig(f"results/plots/{name}_{seqlen}.png", dpi=200)
            curves.to_excel(f"results/curvedata/{name}_{seqlen}.xlsx")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--run-id', required=False, type=str, default=None)
@click.option("--lod", is_flag=True, help="Visualize LOD")
def main(input_filepath, run_id=None, lod=False):
    if lod:
        exp_name = "torch-silver-foam-lod"
    else:
        exp_name = "torch-silver-foam-multilabel-classification"
    if run_id is None:
        run_id = get_last_run_id(exp_name)

    path = f'runs:/{run_id}/eval/summary_per_slen.pickle'
    newpath = mlflow.artifacts.download_artifacts(path)
    with open(newpath, "rb") as f:
        data = pickle.load(f)

    with open(f"{input_filepath}/mapping_dict.pkl", 'rb') as f:
        mapping = pickle.load(f)

    prfx = os.path.commonprefix(list(mapping.values()))
    mapping = {k: v.replace(prfx, "") for k, v in mapping.items()}

    shutil.rmtree("results", ignore_errors=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/curvedata", exist_ok=True)
    os.makedirs("results/av_preds", exist_ok=True)

    micro_acc = make_acc_per_sample(data, mapping)
    make_average_pred(data)
    make_curves(data)
    total_f1 = make_summary_metrics(data)

    mlflow.start_run(run_id=run_id)
    mlflow.log_artifacts("results", "results")
    mlflow.log_metric("micro_acc", micro_acc)
    mlflow.log_metric("total_f1", total_f1)
    mlflow.end_run()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
