import os
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2 as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm



def main(config: dict[str]):
    mlflow.set_tracking_uri('http://localhost:3113')

    # Load the model
    model = mlflow.pytorch.load_model(config["model_path"]).cuda()

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

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        for images, labels in tqdm(dataloader):
            images: torch.Tensor = images.cuda()
            labels: np.ndarray = labels.numpy()

            # Forward pass
            outputs: torch.Tensor = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

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

    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")

    config_path = parser.parse_args().config
    with open(config_path, "r") as f:
        config = f.read()

    configs = json.loads(config)
    for config in configs:
        print("Evaluating with model:", config["model_name"])
        main(config)
