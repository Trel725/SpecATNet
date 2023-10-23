import logging
import os
import pickle
from pathlib import Path
import re

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv


def norm_func(x):
    def std_norm(x):
        return (x - x.mean()) / x.std()
    return np.apply_along_axis(std_norm, 1, x)


def main_data_filter(fname):
    neg_kwds = ["substrate", "lod", "conc"]
    if any([i.lower() in fname.lower() for i in neg_kwds]):
        return False
    return True


# def main_data_filter(fname):
#     neg_kwds = ["substrate", "lod", "conc", "2_component_mixtures"]
#     if any([i.lower() in fname.lower() for i in neg_kwds]):
#         return False
#     return True

def bad_data_filter(fname):
    neg_kwds = ["substrate", "lod", "conc"]
    pos_kwds = ["2_component_mixtures",
                'PET_mixtures_to_balance_data', "Simulated_and_wastewater"]
    if any([i.lower() in fname.lower() for i in neg_kwds]):
        return False

    if any(k.lower() in fname.lower() for k in pos_kwds):
        return True
    return False


def lod_filter(fname):
    neg_kwds = ["LoD_NYLON", "PE", "PET", "PMMA", "PTFE"]
    pos_kwds = ["lod", 'conc', "negativ"]
    if "Concentration_dependence" in fname:
        return True
    if any([i.lower() in fname.lower() for i in neg_kwds]):
        return False

    if any(k in fname.lower() for k in pos_kwds):
        return True
    return False


def rest_filter(fname):
    if main_data_filter(fname) or lod_filter(fname):
        return False
    return True


def parse_string(string, polymers):
    regex = "[^a-z]{}[^a-z]"
    res = []
    for poly in polymers:
        pat = regex.format(poly.lower())
        res.append(bool(re.findall(pat, string.lower())))
    return res


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--filter', type=click.Choice(['main_data', 'lod', 'rest']), required=False, default=None)
def main(input_filepath, output_filepath, filter):
    df = pd.read_parquet(input_filepath + "/preprocessed.parquet")
    with open(input_filepath + "/mapping_dict.pkl", "rb") as f:
        mapping_dict = pickle.load(f)

    prfx = os.path.commonprefix(list(mapping_dict.values()))
    mapping_dict = {k: v.replace(prfx, "") for k, v in mapping_dict.items()}
    mapping_dict = {k: v.replace(
        "Concentration_dependence", "Concentration_dependence_PS") for k, v in mapping_dict.items()}

    if filter is None:
        filters = ['main_data', 'lod', 'rest']
    else:
        filters = [filter]

    for filter in filters:
        polymers = ["PE", "PMMA", "PTFE", "PS", "Nylon", "PET"]
        if filter == 'lod':
            polymers = ["PS"]

        filter_to_use = globals()[f"{filter}_filter"]
        poly_mapping = {k: parse_string(v, polymers=polymers)
                        for (k, v) in mapping_dict.items()
                        if filter_to_use(v)}
        Y = df['key'].apply(lambda x: poly_mapping[x]
                            if x in poly_mapping else None).dropna()
        Y = pd.DataFrame(Y.to_list(), columns=polymers, index=Y.index)
        samples = df.loc[Y.index]['key']
        logging.info(
            f"Filter {filter}: selected {samples.nunique()} samples from {df['key'].nunique()}")
        for s in [v for (k, v) in mapping_dict.items() if filter_to_use(v)]:
            print(s)

        X = df.loc[Y.index].drop(columns="key")
        X = pd.DataFrame(norm_func(X))

        X.columns = X.columns.astype(str)
        path = output_filepath + f"/preselected/{filter}"
        os.makedirs(path, exist_ok=True)
        X.to_parquet(path + "/X.parquet")
        Y.to_parquet(path + "/Y.parquet")
        pd.DataFrame(samples).to_parquet(path + "/samples.parquet")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
