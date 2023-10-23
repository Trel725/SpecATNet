import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data_utils import make_k_fold
from models import DenseNet, SeqNet, SpecATNet, AlexNet, ResNet, DiscrimNet, TemplateNet
from sam import SAM
from evaluation import accuracy
from common import device


def training_loop(X, Y, samples, oversample=False, random_state=42):
    models = {}
    for train_loader, test_loader, split_idx in make_k_fold(X, Y, samples, oversample=oversample, random_state=random_state):
        loss_fn = nn.BCEWithLogitsLoss()
        model = torch.load("model.tmp").to(device)
        optimizer = SAM(model.parameters(), torch.optim.AdamW,
                        lr=1e-3, weight_decay=3)

        mlflow.log_params(optimizer.defaults)
        mlflow.log_param("optimizer_name", optimizer.__class__.__name__)
        steps = len(train_loader)
        val_steps = len(test_loader)
        n_epochs = 1
        mlflow.log_param("n_epochs", n_epochs)
        for epoch in range(n_epochs):
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
                    print(f"acc: {acc:.3f}")
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

    for idx, model in models.items():
        mlflow.pytorch.log_model(model, f"model_{idx}")
    return models


def load_data(input_filepath, load_templates=False):
    X = pd.read_parquet(input_filepath + "/X.parquet")
    Y = pd.read_parquet(input_filepath + "/Y.parquet")
    X.index = Y.index
    samples = pd.read_parquet(input_filepath + "/samples.parquet")
    if load_templates:
        templates = pd.read_parquet(
            input_filepath + "/../../templates_resampled.parquet").values.astype(np.float32).T
        return X, Y, samples['key'], templates
    else:
        return X, Y, samples['key']


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option("-m", '--model', type=click.Choice(['alexnet', 'resnet', 'densenet']), required=True, help="Model to use")
@click.option('-a', '--attention', is_flag=True, help="Use attention-average")
@click.option('-o', '--oversample', is_flag=True, help="Oversample during training")
def main(input_filepath, model, attention, oversample):
    if "lod" in input_filepath:
        mlflow.set_experiment("torch-silver-foam-lod")
    else:
        mlflow.set_experiment("torch-silver-foam-multilabel-classification")
    mlflow.start_run()

    X, Y, samples = load_data(input_filepath)
    N_CLASSES = Y.shape[1]
    logging.info("Data loaded successfully")

    mlflow.log_param("attention", attention)
    mlflow.log_param("model", model)
    seed = 22670  # np.random.randint(0, 100000)
    models = dict(densenet=DenseNet, alexnet=AlexNet, resnet=ResNet)
    Model = models[model]

    params = dict(conv_channels=[7, 17, 45],
                  kern_sizes=(7, 15, 31),
                  droprate=0.2,
                  act=nn.LeakyReLU(),
                  norm='batch',
                  stride=1)

    if attention:
        params["out_shape"] = 64
        dnet = Model(**params)
        # seqnet = DiscrimNet(dnet, embed_dim=64, logits=N_CLASSES).to(device)
        # seqnet = TemplateNet(dnet, templates, logits=N_CLASSES).to(device)
        seqnet = SpecATNet(dnet, embed_dim=64, logits=N_CLASSES).to(device)
    else:
        params["out_shape"] = N_CLASSES
        dnet = Model(**params)
        seqnet = SeqNet(dnet).to(device)

    mlflow.log_params(params)
    mlflow.log_param("seed", seed)
    torch.save(seqnet, "model.tmp")
    models = training_loop(
        X, Y, samples, oversample=oversample, random_state=seed)

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
