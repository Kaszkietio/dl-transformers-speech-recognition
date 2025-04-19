import sys

sys.path.append("./src")

import os
import json
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from tqdm import tqdm

from models.simple_transformer import SimpleTransformer
from utils import set_seed, get_device

MODELS = {
    "SimpleTransformer": SimpleTransformer,
}

OPTIMIZERS = {
    "SGD": SGD,
    "AdamW": AdamW
}

SCHEDULERS = {
    "ExponentialLR": ExponentialLR,
    "LinearLR": LinearLR,
    "SequentialLR": SequentialLR,
    "ConstantLR": ConstantLR
}


def get_datasets(data_path: str, batch_size: int):
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    ds_train = ImageFolder(train_path, transform=T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.Resize((116, 116)),
        T.ToDtype(torch.float32, scale=True),
    ]))
    ds_valid = ImageFolder(valid_path, transform=T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.Resize((116, 116)),
        T.ToDtype(torch.float32, scale=True),
    ]))

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return loader_train, loader_valid


def train(
    model: nn.Module,
    train_ds: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss
):
    losses = []
    accuracies = []
    batch_sizes = []

    num_classes = len(train_ds.dataset.classes)

    for input, target in tqdm(train_ds):
        optimizer.zero_grad()

        # print("Input shape:", input.shape)
        # print("Target shape:", target.shape)

        input, target = input.cuda(), target.cuda()
        ohe_target = F.one_hot(target, num_classes).type(torch.float32).reshape(-1, num_classes)
        output = model(input)
        # print("Output shape:", output.shape)
        # print("OHE Target shape:", ohe_target.shape)
        loss = criterion(output, ohe_target)

        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, dim=-1)
        accurate_pred = (pred == target).type(torch.float32)
        accuracies.append(torch.mean(accurate_pred).item())

        losses.append(loss.item())
        batch_sizes.append(len(input))

    accuracy = np.average(accuracies, weights=batch_sizes)
    loss = np.average(losses, weights=batch_sizes)
    return accuracy, loss


def evaluate(
    model: nn.Module,
    valid_ds: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    val_losses = []
    val_accuracies = []
    batch_sizes = []

    num_classes = len(valid_ds.dataset.classes)

    with torch.no_grad():
        for input, target in tqdm(valid_ds):
            input, target = input.cuda(), target.cuda()
            ohe_target = F.one_hot(target, num_classes).type(torch.float32).reshape(-1, num_classes)
            output = model(input)
            val_loss = criterion(output, ohe_target)

            # Calculate accuracy
            pred = torch.argmax(output, dim=-1)
            accurate_pred = (pred == target).type(torch.float32)
            val_accuracies.append(torch.mean(accurate_pred).item())

            val_losses.append(val_loss.item())
            # Store information regarding
            batch_sizes.append(len(input))

    val_accuracy = np.average(val_accuracies, weights=batch_sizes)
    val_loss = np.average(val_losses, weights=batch_sizes)
    return val_accuracy, val_loss


def main(config: dict):
    print("Starting training:")

    experiment_name = config["experiment_name"]
    print("Experiment name:", experiment_name)

    # Set seed for reproducibility
    seed = int(config["seed"]) if "seed" in config else 0
    set_seed(seed)
    print("Setting seed:", seed)

    device = get_device()
    print("Device: ", device)

    checkpoint = config["checkpoint_folder"]
    checkpoint = os.path.abspath(checkpoint)
    os.makedirs(checkpoint, exist_ok=True)
    print("Checkpoint folder:", checkpoint)

    # Save config to checkpoint folder
    with open(os.path.join(checkpoint, "config.json"), "w") as f:
        f.write(json.dumps(config))

    batch_size = int(config["batch_size"]) if "batch_size" in config else 256
    print("Batch size:", batch_size)
    data_path = os.path.abspath(config["data_path"])
    print("Data path:", data_path)
    spec_train, spec_valid = get_datasets(data_path, batch_size)

    model: nn.Module = MODELS[config["model"]](**config["model_params"]).to(device)
    print("Model:", config["model"])
    print("Model params:", config["model_params"])
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["optimizer"]](params=model.parameters(),
                                                **config["optimizer_params"])
    print("Optimizer:", optimizer)

    if "scheduler" in config:
        scheduler_params = config["scheduler_params"]
        if config["scheduler"] == "SequentialLR":
            inner_schedulers = scheduler_params["schedulers"]
            schedulers = [SCHEDULERS[inner_scheduler](optimizer, **params)
                                for inner_scheduler, params in inner_schedulers.items()]
            print("scheduler params", scheduler_params)
            scheduler = SequentialLR(optimizer,
                                     schedulers=schedulers,
                                     milestones=scheduler_params["milestones"])
        else:
            scheduler = SCHEDULERS[config["scheduler"]](**scheduler_params)

    print("Scheduler:", scheduler)
    criterion = nn.CrossEntropyLoss()

    epochs = int(config["epochs"])
    print("Number of epochs: ", epochs)
    warmup_epochs = int(config["warmup_epochs"])
    print("Warmup epochs:", warmup_epochs)

    min_delta = float(config["early_stopping"]["min_delta"]) if "early_stopping" in config else 0.0
    print("Min delta:", min_delta)
    patience = int(config["early_stopping"]["patience"]) if "early_stopping" in config else 0
    print("Patience:", patience)

    classes = spec_train.dataset.classes
    classes = sorted(classes, key=lambda x: spec_train.dataset.class_to_idx[x])

    best_model = model.state_dict()
    best_loss = float("inf")
    best_loss_epoch = 0

    previous = {"loss": float("inf"), "accuracy": 0.0, "val_loss": float("inf"), "val_accuracy": 0.0}

    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:3113")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        mlflow.log_param("model", config["model"])
        mlflow.log_param("model_params", config["model_params"])
        mlflow.log_param("optimizer", config["optimizer"])
        mlflow.log_param("optimizer_params", config["optimizer_params"])
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("scheduler", config["scheduler"])
        mlflow.log_param("scheduler_params", config["scheduler_params"])
        mlflow.log_params(config["model_params"])
        mlflow.log_params(config["optimizer_params"])
        mlflow.log_params(config["scheduler_params"])

        for epoch in range(epochs):
            print("Epoch", epoch)

            print("Processing training")
            model.train()
            accuracy, loss = train(model, spec_train, optimizer, criterion)

            print("Processing validation")
            model.eval()
            val_accuracy, val_loss = evaluate(model, spec_valid, criterion)

            scheduler.step()

            print(f"Epoch {epoch} finished")
            print(f"Loss: {loss:.4f}({(loss - previous['loss']):.4e})", end=' ')
            print(f"Accuracy: {accuracy:.4f}({(accuracy - previous['accuracy']):.4e})", end=' ')
            print(f"Validation Loss: {val_loss:.4f}({(val_loss - previous['val_loss']):.4e})", end=' ')
            print(f"Validation Accuracy: {val_accuracy:.4f}({(val_accuracy - previous['val_accuracy']):.4e})")
            print()

            previous["loss"] = loss
            previous["accuracy"] = accuracy
            previous["val_loss"] = val_loss
            previous["val_accuracy"] = val_accuracy

            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0])


            # Saving checkpoint
            if epoch >= warmup_epochs and min_delta < best_loss - val_loss:
                best_model = model.state_dict()

                artifact_path = f"model_checkpoint_epoch_{epoch}"
                mlflow.pytorch.log_model(model, artifact_path=artifact_path)

            # Early stopping
            if min_delta < best_loss - val_loss:
                best_loss_epoch = epoch
                best_loss = val_loss
            elif epoch - best_loss_epoch >= patience:
                print("Early stopping!")
                break

        model.load_state_dict(best_model)
        model.eval()
        X, _ = next(iter(spec_valid))
        X = X.cuda()
        signature = mlflow.models.infer_signature(X.detach().cpu().numpy(), model(X).detach().cpu().numpy())
        artifact_path = f"model_final_epoch_{best_loss_epoch}"
        mlflow.pytorch.log_model(model, artifact_path, signature=signature)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = f.read()
    config = json.loads(config)

    main(config)