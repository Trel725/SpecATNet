# -*- coding: utf-8 -*-
import glob
import logging
import shutil
import subprocess
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

parent = str(Path(__file__).resolve().parent)

def combine_all(root, outpath):
    files = glob.glob(root + "/**/*.csv", recursive=True)
    print(files)
    subprocess.run([f"{parent}/universal_preprocess.py", 
                    "--resample",
                    "--wvl", "300", "1500", "500",
                    "--int-thresh",  "100", 
                    "-t 20000", 
                    "-s", outpath,
                    "-d", *files])


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Joining input to single big csvs...")
    combine_all(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
