import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import mlflow
from train_model import load_data
from evaluation import summarize_per_seqlens
import glob


def get_last_models(exp_name, run_id=None):
    exps = mlflow.search_runs(experiment_names=[exp_name])
    if run_id is None:
        last_run = exps.sort_values(by='end_time').iloc[-1]
    else:
        last_run = exps.loc[exps.run_id == run_id].iloc[0]
    if last_run.size == 0:
        raise ValueError("Can't find experiment!")
    models = {}
    for model_path in glob.glob(last_run.artifact_uri[7:] + "/model*"):
        idx = int(model_path.split("_")[-1])
        model = mlflow.pytorch.load_model(
            last_run.artifact_uri + f"/model_{idx}")
        models[idx] = model
    return models, last_run.run_id


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--run-id', required=False, type=str, default=None)
@click.option("--lod", is_flag=True, help="Evaluate LOD")
def main(input_filepath, run_id, lod):
    if lod:
        exp_name = "torch-silver-foam-lod"
    else:
        exp_name = "torch-silver-foam-multilabel-classification"
    X, Y, samples = load_data(input_filepath)
    models, run_id = get_last_models(exp_name, run_id)
    run = mlflow.get_run(run_id=run_id)
    random_state = int(run.data.params['seed'])
    mlflow.start_run(run_id=run_id)
    if lod:
        seqlens = range(1, 21)
    else:
        seqlens = [4, 8, 12, 16, 20, 26, 32, 64, 96]
    eval_results = summarize_per_seqlens(
        X, Y, samples, models, seqlens=seqlens, random_state=random_state)
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
