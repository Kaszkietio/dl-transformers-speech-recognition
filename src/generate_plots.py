import argparse
import mlflow
import json
import matplotlib.pyplot as plt

def fetch_metrics_from_mlflow(run_id, metric_name):
    """
    Fetches the specified metric from an MLflow run.
    """
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    return [metric.value for metric in metric_history], [metric.step for metric in metric_history]

def plot_metrics(models, metrics, title, line_styles):
    """
    Plots the specified metrics for the given models.
    """
    for model_name, run_id in models:
        for metric_name, line_style in zip(metrics, line_styles):
            metric_values, steps = fetch_metrics_from_mlflow(run_id, metric_name)
            plt.plot(steps, metric_values, line_style, label=f"{model_name} - {metric_name}")

    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="Generate plots for model metrics from MLflow.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (JSON format).")
    args = parser.parse_args()

    # Load configuration from the JSON file
    with open(args.config, "r") as config_file:
        config = json.load(config_file)


    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

    for plot in config["plots"]:
        models = [(model["name"], model["run_id"]) for model in plot["models"]]

        plt.subplots(1, 2, figsize=(15, 6))
        plt.suptitle("Model Metrics Comparison")

        # Plot loss and val_loss
        plt.subplot(1, 2, 1)
        plot_metrics(
            models,
            metrics=["loss", "val_loss"],
            title="Loss and Validation Loss",
            line_styles=["--", "-"]
        )

        # Plot accuracy and val_accuracy
        plt.subplot(1, 2, 2)
        plot_metrics(
            models,
            metrics=["accuracy", "val_accuracy"],
            title="Accuracy and Validation Accuracy",
            line_styles=["--", "-"]
        )

        plt.savefig(plot["output_path"])
        plt.close()

if __name__ == "__main__":
    main()