import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
import torch.utils.data as tdata
import cvxpy as cp
from raman_dataset import RamanSpectraSeqDataset, NMFRamanSpectraSeqDataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

Dataset = RamanSpectraSeqDataset


def unpickle(file):
    with open(file, "rb") as f:
        res = pickle.load(f)
    return res


def train_test_split_samples(X, Y, samples, random_state=42):
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    un_samples = np.unique(samples)
    for split_idx, (train_idx, test_idx) in enumerate(kfold.split(un_samples)):
        samples_train = un_samples[train_idx]
        samples_test = un_samples[test_idx]
        tr_mask = samples.apply(lambda x: x in samples_train)
        yield (X.loc[tr_mask], X.loc[~tr_mask],
               Y.loc[tr_mask], Y.loc[~tr_mask],
               samples.loc[tr_mask], samples.loc[~tr_mask],
               split_idx)


def make_k_fold(X, Y, samples, seq_len=32, batch_size=64, oversample=False, random_state=42):
    for (X_train, X_test,
         Y_train, Y_test,
         samples_train, samples_test,
         split_idx) in train_test_split_samples(X, Y, samples, random_state=random_state):
        print(f"Split idx {split_idx}")
        if oversample:
            X_train, Y_train, samples_train = oversample_df(
                X_train, Y_train, samples_train)
        rsd_train = Dataset(X_train,
                            Y_train.astype(float),
                            samples_train,
                            seq_len=seq_len, min_seq_len=8)
        rsd_test = Dataset(X_test,
                           Y_test.astype(float),
                           samples_test,
                           seq_len=seq_len, min_seq_len=8)

        train_loader = tdata.DataLoader(
            rsd_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = tdata.DataLoader(
            rsd_test, batch_size=batch_size, shuffle=True, num_workers=4)
        yield train_loader, test_loader, split_idx


def oversample_df_smote(X, Y, samples):
    cnts = {k: np.minimum(v, 500)
            for k, v in samples.value_counts().to_dict().items()}
    unders = RandomUnderSampler(sampling_strategy=cnts)
    undersampled_X, undersampled_samples = unders.fit_resample(X, samples)
    overs = SMOTE()
    X_ov, samples_ov = overs.fit_resample(undersampled_X, undersampled_samples)
    label_per_sample = Y.groupby(samples).agg(
        lambda x: x.iloc[0]).to_dict(orient='index')
    Y_ov = samples_ov.apply(lambda x: pd.Series(label_per_sample[x]))
    return X_ov, Y_ov, samples_ov


def oversample_df(X, Y, samples):

    Y_samples = Y.copy()
    Y_samples['samples'] = samples
    Y_uniq = Y_samples.drop_duplicates()
    alltogether = X.copy()
    alltogether[Y.columns] = Y
    alltogether['samples'] = samples

    A = Y_uniq.drop(columns="samples").values.astype(int).T
    b = np.array([0.5]*Y.shape[1]) * A.shape[1]
    # Construct a CVXPY problem
    x = cp.Variable(A.shape[1], integer=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - b)) + \
        0.1 * cp.Minimize(cp.max(x))
    prob = cp.Problem(objective, [cp.max(x) <= 5, cp.min(x) >= 1])
    loss = prob.solve(solver=cp.SCIP)
    print("SCIP loss:", loss)
    solution = list(prob.solution.primal_vars.values())[0]
    values = Y_uniq['samples'].values
    counts = solution
    to_oversample = dict(
        zip(values[np.where(counts > 1)[0]], counts[np.where(counts >= 1.9)[0]]))
    print(to_oversample)
    to_extend = []
    for sample, num_over in to_oversample.items():
        to_extend.extend(
            [alltogether.loc[alltogether['samples'] == sample]]*int(num_over - 1))

    to_extend_labelled = []
    for idx, overs_data in enumerate(to_extend):
        overs_data['samples'] = idx
        to_extend_labelled.append(overs_data.copy())

    ov_smpls = pd.concat(to_extend_labelled)
    X_over = pd.concat([X.reset_index(drop=True),
                        ov_smpls.drop(columns=[*Y.columns, "samples"]).reset_index(drop=True)])
    Y_over = pd.concat([Y.reset_index(drop=True),
                        ov_smpls[Y.columns].reset_index(drop=True)])

    smpls_extended = ov_smpls['samples'].reset_index(
        drop=True) + samples.max() + 1
    samples_over = pd.concat([samples.reset_index(drop=True), smpls_extended])
    return X_over, Y_over, samples_over
