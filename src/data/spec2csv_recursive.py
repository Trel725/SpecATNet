#!/usr/bin/env python3

import sys
import spc
import glob
import os
import subprocess
import argparse
import pandas as pd
from os.path import isfile
from jwslib import read_file as read_jws
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="Root of directory to work with")

parser.add_argument("-f", "--format",
                    choices=['spc', 'spa', 'txt_cvut', "txt_cvut_map", "jws"],
                    required=True)

parser.add_argument("-d", "--detect-existence", action='store_true',
                    help="Prevent overwriting of already converted filed")
parser.add_argument("--delete", action='store_true',
                    help="Delete SPC after converting")
args = parser.parse_args()
# usage:


def convert_spa(fname, res_filename):
    fname = os.path.abspath(fname)
    path, basename = os.path.split(fname)
    #print("-"*10, path, "-"*10)
    print(path)
    subprocess.run(["spa2csv.rb", fname], cwd=path)
    # os.system("spa-reader {}".format(fname.replace(" ", "\\ ")))
    # os.rename(path + os.sep + default_fname, fname[:-4] + ".csv")


def convert_spc(fname, res_filename):
    s = spc.File(fname)
    s.write_file(res_filename)


def convert_jws(fname, res_filename):
    status, header, y = read_jws(fname)
    y = np.array(y).flatten()
    x = np.arange(header.x_for_first_point, header.x_for_last_point + header.x_increment, header.x_increment)
    pd.DataFrame(dict(x=x, y=y)).to_csv(res_filename, index=None,
                                        header=None, sep="\t")


def convert_txt_cvut(fname, res_filename):
    data = pd.read_csv(fname, sep="\t",
                       header=None)
    data.to_csv(res_filename, index=None,
                header=None, sep="\t")


def convert_txt_cvut_map(fname, res_filename):
    data = pd.read_csv(fname, sep="\t+")
    for idx, (_, df) in enumerate(data.groupby(["#X", "#Y"])):
        df = df[["#Wave", "#Intensity"]]
        f = res_filename.replace(".csv", "")
        df.to_csv(f"{f}_{idx}.csv", index=None,
                    header=None, sep="\t")

extensions = {
    "spc": ".spc",
    "spa": ".spa",
    "txt_cvut": ".txt",
    "txt_cvut_map": "txt",
    "jws": ".jws"
}

converters = {
    "spc": convert_spc,
    "spa": convert_spa,
    "txt_cvut": convert_txt_cvut,
    "txt_cvut_map": convert_txt_cvut_map,
    "jws": convert_jws
}

print(f"{args.root}/**/*{extensions[args.format]}")

for file in glob.iglob(f"{args.root}/**/*{extensions[args.format]}", recursive=True):
    print("Working with", file)
    res_filename = os.path.splitext(file)[0] + ".csv"
    if args.detect_existence:
        if isfile(res_filename):
            continue
    try:
        converters[args.format](file, res_filename)
        if args.delete:
            os.remove(file)
    except Exception as e:
        print("Can't convert file {}, {}".format(file, str(e)))
