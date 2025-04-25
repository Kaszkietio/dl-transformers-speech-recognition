import sys

sys.path.append(".\src")

import os
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, ConfusionMatrixDisplay
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2 as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from final_model import FinalModel

def main(config: dict[str]):
    mlflow.set_tracking_uri('http://localhost:3113')

    # Load the model
    base_10 = mlflow.pytorch.load_model(config["base_10_path"]).cuda()
    known_unknown = mlflow.pytorch.load_model(config["known_unknown_path"]).cuda()
    voice_silence = mlflow.pytorch.load_model(config["voice_silence_path"]).cuda()
    model = FinalModel(base_10, known_unknown, voice_silence).cuda()

    # Load the dataset
    dataset = ImageFolder(root=config["dataset_path"], transform=T.Compose([
        T.Resize((116, 116)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]))
    # num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)

    all_predictions = []
    all_labels = []

    base_10_classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']


    # Evaluate the model
    with torch.no_grad():
        model.eval()
        for images, labels in tqdm(dataloader):
            images: torch.Tensor = images.cuda()
            labels: np.ndarray = labels.numpy()

            # Forward pass
            voice_silence_out, known_unknown_out, base_10_out = model(images)
            voice_silence_pred = torch.argmax(voice_silence_out, dim=1).cpu().numpy()
            known_unknown_pred = torch.argmax(known_unknown_out, dim=1).cpu().numpy()
            base_10_pred = torch.argmax(base_10_out, dim=1).cpu().numpy()

            # Convert base_10_pred to base_10 classes
            preds = np.ones_like(base_10_pred)
            for i in range(len(base_10_pred)):
                preds[i] = dataset.class_to_idx[base_10_classes[base_10_pred[i]]]

            preds = np.where(known_unknown_pred == 1, dataset.class_to_idx["unknown"], preds)
            preds = np.where(voice_silence_pred == 0, dataset.class_to_idx["silence"], preds)

            all_predictions.extend(preds)
            all_labels.extend(labels)

    labels = np.array(all_labels)
    preds = np.array(all_predictions)

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None,
                                                                     labels=list(range(len(dataset.classes))))

    cm_display = ConfusionMatrixDisplay.from_predictions(labels, preds,
                                                         display_labels=dataset.classes,
                                                         cmap="Blues", normalize=None)
    os.makedirs(config["output_path"], exist_ok=True)
    cm_display.figure_.savefig(os.path.join(config["output_path"], "confusion_matrix.png"))

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Support:", support)

    # Save metrics to CSV
    df = pd.DataFrame({
        "Class": dataset.classes,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Support": support
    })
    df.to_csv(os.path.join(config["output_path"], "metrics.csv"), index=False)




if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate a final model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")

    config_path = parser.parse_args().config
    with open(config_path, "r") as f:
        config = f.read()

    config = json.loads(config)
    print("Evaluating with model:", config["model_name"])
    main(config)
