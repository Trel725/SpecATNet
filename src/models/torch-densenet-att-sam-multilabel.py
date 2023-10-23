import pickle

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchinfo
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import check_random_state
from tqdm.auto import tqdm

from .data_utils import make_k_fold
from .models import DenseNet, SeqNet, SpecATNet
from .sam import SAM


def spickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# !export SCIPOPTDIR=/home/user/SCIPOptSuite-8.0.2-Linux


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


mlflow.set_experiment("torch-silver-foam-multilabel-classification")
run = mlflow.start_run(run_name="att sequence var length input")
mlflow.set_tag("balancing", "None")
mlflow.set_tag("preproc", "norm only + snr filtering (threshold 0.03)")
mlflow.set_tag("samplewise splitting", "true")


def randrows(A, size):
    return A[np.random.randint(A.shape[0], size=size), :]


USE_ATTENTION = mlflow.log_param("attention", True)
OVERSAMPLE = mlflow.log_param("oversample", True)

if USE_ATTENTION:
    params = dict(conv_channels=[7, 17, 45],
                  kern_sizes=(7, 15, 31), out_shape=64,
                  droprate=0.4,
                  act=torch.tanh)

    dnet = DenseNet(X.shape[1], **params)
    seqnet = SpecATNet(dnet, embed_dim=64, logits=5).to(device)
else:
    params = dict(conv_channels=[7, 17, 45],
                  kern_sizes=(7, 15, 31), out_shape=5,
                  droprate=0.4,
                  act=torch.tanh)

    dnet = DenseNet(X.shape[1], **params)
    seqnet = SeqNet(dnet).to(device)

mlflow.log_params(params)

torchinfo.summary(seqnet, input_data=torch.Tensor(X.values[None, :12, None, :]).to(device))





torch.save(seqnet, "model.tmp")


def accuracy(pred, true):
    return ((pred > 0.5) == true).float().mean()


models = {}
for train_loader, test_loader, split_idx in make_k_fold(X, Y, samples):
    loss_fn = nn.BCEWithLogitsLoss()
    model = torch.load("model.tmp").to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1, lr=1e-3)
    optimizer = SAM(model.parameters(), torch.optim.AdamW,
                    lr=1e-3, weight_decay=1)

    mlflow.log_params(optimizer.defaults)
    mlflow.log_param("optimizer_name", optimizer.__class__.__name__)
    steps = len(train_loader)
    val_steps = len(test_loader)
    step_every = 5

    for epoch in range(mlflow.log_param("n_epochs", 1)):
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            def closure():
                loss = loss_fn(model(inputs), labels)
                loss.backward()
                return loss

            loss = loss_fn(outputs := model(inputs), labels,)
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()

            running_loss += loss.item()
            with torch.no_grad():
                acc = accuracy(outputs, labels)
            running_acc += acc.item()
            train_loader.dataset.on_batch_end()

        # now evaluate
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()
                val_acc += accuracy(outputs, labels).item()
                test_loader.dataset.on_batch_end()

        print(f'[{epoch}] loss: {running_loss / steps:.3f} acc: {running_acc / steps:.3f} \
    val_loss: {val_loss / val_steps:.3f} val_acc: {val_acc / val_steps:.3f}')
        mlflow.log_metrics(dict(loss=running_loss / steps, acc=running_acc / steps,
                                val_loss=val_loss / val_steps, val_acc=val_acc / val_steps), epoch)

    models[split_idx] = model


def _create_batch(arr, slen):
    num_splits = np.floor(arr.shape[0] / slen)
    return np.array(np.split(arr[:int(num_splits*slen)], num_splits))


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


accs_per_slen = {}
average_preds = {}
for slen in tqdm(range(1, 21)):
    evaluation_results = []
    av_pr = []
    for (X_train, X_test,
         Y_train, Y_test,
         samples_train, samples_test,
         split_idx) in train_test_split_samples(X, Y, samples):
        preds = predict_by_group(X_test, samples_test, model, slen=slen)
        ev = evaluate_preds(Y_test, samples_test, preds)
        evaluation_results.append(ev)
        av_pr.append(pd.DataFrame({k: (v > 0.5).mean(axis=0)
                     for k, v in preds.items()}).T)
    accs_per_slen[slen] = pd.concat(evaluation_results).sort_index()
    average_preds[slen] = pd.concat(av_pr).sort_index()


for idx, model in models.items():
    mlflow.pytorch.log_model(model, f"model_{idx}")


accs_sum_df = pd.concat([i for i in accs_per_slen.values()], axis=1)

accs_sum_df['sample'] = accs_sum_df.index

accs_sum_df['sample'] = accs_sum_df['sample'].apply(lambda x: mapping[x].split("/")[-1])


accs_sum_df.sort_values(by=19).iloc[:20]


accs_sum_df.to_csv("accs_per_sample.csv", index=None)
mlflow.log_artifact("accs_per_sample.csv", "final_metrics")


spickle(average_preds, "average_preds.pkl")
mlflow.log_artifact("average_preds.pkl", "final_metrics")


with pd.ExcelWriter("average_preds.xlsx") as writer:
    for slen, av_pr in average_preds.items():
        av_pr['sample'] = av_pr.index
        av_pr['sample'] = av_pr['sample'].apply(
            lambda x: mapping[x].split("/")[-1])
        av_pr.columns = [*Y.columns, "sample"]
        av_pr.to_excel(writer, sheet_name=f"Seq len {slen}")

mlflow.log_artifact("average_preds.xlsx", "final_metrics")


metrics = {}
for train_loader, test_loader, sample_to_hold in make_k_fold(X, Y, samples):
    model = models[sample_to_hold]
    metrics[sample_to_hold] = {}
    for slen in tqdm(range(1, 21)):
        test_loader.dataset.curr_seq_len = slen
        stats = dict(true=[], pred=[])
        with torch.no_grad():
            accs = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                pred = model(inputs)
                accs.append(accuracy(pred, labels).item())
                stats['true'].append(labels.detach().cpu().numpy())
                stats['pred'].append(pred.detach().cpu().sigmoid().numpy())
        metrics[sample_to_hold][slen] = dict(acc=np.mean(accs),
                                             true=np.vstack(stats['true']),
                                             pred=np.vstack(stats['pred']))


for sample_to_hold, m in metrics.items():
    plt.plot([k for k in m], [m[k]['acc']
             for k in m], label=f"{sample_to_hold}")

plt.legend()
plt.xlabel("Sequence length")
plt.ylabel("Accuracy")


# In[ ]:


cross_val_acc = np.array([[m[k]['acc'] for k in m]
                         for m in metrics.values()]).mean()
mlflow.log_metric("cross_val_acc", cross_val_acc)


res = dict()  # dict(zip(Y.columns, [[]]*Y.columns.size))
for split_idx, metric in metrics.items():
    for slen_, m in metric.items():
        if slen_ not in res:
            res[slen_] = []
        per_poly_res = {}
        for i in range(Y.shape[1]):
            t = sklearn.metrics.precision_recall_fscore_support(m['true'][:, i],
                                                                m['pred'][:, i] > 0.5)
            per_poly_res[Y.columns[i]] = [i[0] for i in t]
        res[slen_].append(per_poly_res)


summary = {}
for midx, mname in enumerate(['Precision', "Recall", "F1-score"]):
    summary[mname] = {}
    fig = plt.figure()
    for poly in Y.columns:
        poly_data = np.array([[l[poly][midx] for l in r]
                             for r in res.values()])
        plt.plot(np.arange(1, 21), poly_data.mean(axis=1), label=poly)
#         plt.title(mname)
        plt.xlabel("Sequence length")
        plt.ylabel(mname)
        plt.tight_layout()
        summary[mname][poly] = poly_data
    plt.legend()
    mlflow.log_figure(fig, f"figures/{mname}.png")
    spickle(summary, "summary.pickle")
    mlflow.log_artifact("summary.pickle", "final_metrics")


# In[ ]:


summary_df = pd.DataFrame({(outerKey, innerKey): values.mean(axis=1)
                           for outerKey, innerDict in summary.items()
                           for innerKey, values in innerDict.items()})

summary_df.index = np.arange(1, 21)
summary_df.to_excel("cross_val_summary.xlsx")
mlflow.log_artifact("./cross_val_summary.xlsx", "final_metrics")


pred_ = np.concatenate([metrics[i][len(metrics[i])]['pred'] for i in metrics])
true_ = np.concatenate([metrics[i][len(metrics[i])]['true'] for i in metrics])


fig = plt.figure(figsize=(4, 4))
curve_dfs = {}
for idx, name in enumerate(Y.columns):
    prec, rec, thres = precision_recall_curve(true_[:, idx], pred_[:, idx])
    df_ = pd.DataFrame(dict(precision=prec[:len(
        thres)], recall=rec[:len(thres)], threshold=thres[:len(thres)]))
    curve_dfs[name] = df_
    plt.plot(prec, rec, label=name)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
plt.legend()
mlflow.log_figure(fig, "figures/prec_rec_curve.png")


with pd.ExcelWriter('prec_rec_curve.xlsx') as writer:
    for name, df_ in curve_dfs.items():
        df_.to_excel(writer, sheet_name=name)

mlflow.log_artifact("prec_rec_curve.xlsx", "final_metrics")

mlflow.end_run()
