#!/usr/bin/env python3

import sys
import spc
import glob
import os
import pandas as pd
import rampy
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="Root of directory to work with")
parser.add_argument("-n", '--no-background-subtract', action="store_true",
                    help='Do not subtract background from the data')
parser.add_argument("-s", "--smoothness", type=float, default=1e6,
                    help="Backgorund smoothness parameter, default=1e6")

args = parser.parse_args()

try:
    os.mkdir(args.root + "/joined")
except:
    print("Directory exists!")
    pass

subtract = True
if args.no_background_subtract:
    subtract = False
smoothness = args.smoothness

for root, dirs, files in os.walk(args.root):

    spectra = []

    if "joined" in [os.path.basename(i) for i in os.path.split(root)]:
        continue
    print("Entering directory ", root)
    for f in files:
        if os.path.splitext(f)[1] == ".csv":
            # print("Entered!", os.path.splitext(f))
            try:
                df = pd.read_csv(root + '/' + f, header=None, sep='\t')
                df.sort_values(by=0, inplace=True)
                if subtract:
                    print("Processing ", root + '/' + f)
                    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
                    subtracted, base = rampy.baseline(x, y, np.array(
                        [[0., x[-1]]]), method="als", lam=smoothness)
                    spectra.append(subtracted[:, 0])
                else:
                    spectra.append(df.values[:, 1])
            except Exception as e:
                print("Cant read or process the file", root + '/' + f)
                print(repr(e))
                pass
                # raise ValueError("Cant read or process the file", root + '/' + f)
    if len(spectra) > 1:

        frame = pd.DataFrame(spectra)
        try:
            frame.loc['x_wvl'] = x
        except Exception as e:
            print(e)
        path = args.root + "/joined/" + root.replace(args.root, "")
        os.makedirs(path, exist_ok=True)
        print("Writing data to: ", path + "/" +
              os.path.split(root)[-1] + ".csv")
        frame.to_csv(path + "/" + os.path.split(root)
                     [-1] + ".csv", index_label="Index")
