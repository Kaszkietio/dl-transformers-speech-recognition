import argparse
import mlflow

import matplotlib.pyplot as plt

def fetch_metrics_from_mlflow(run_id, metric_name):
    """
    Fetches the specified metric from an MLflow run.
    """
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    return [metric.value for metric in metric_history], [metric.step for metric in metric_history]

def plot_metrics(models, metric_name):
    """
    Plots the specified metric for the given models.
    """
    plt.figure(figsize=(10, 6))
    for model_name, run_id in models:
        metric_values, steps = fetch_metrics_from_mlflow(run_id, metric_name)
        plt.plot(steps, metric_values, label=model_name)

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison Across Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate plots for model metrics from MLflow.")
    parser.add_argument("--models", type=str, required=True,
                        help="List of models and their run IDs in the format: model1:run_id1,model2:run_id2,...")
    parser.add_argument("--metric", type=str, required=True,
                        help="The name of the metric to plot.")
    args = parser.parse_args()

    # Parse models and run IDs
    models = [tuple(model.split(":")) for model in args.models.split(",")]

    mlflow.set_tracking_uri("http://localhost:3113")

    # Generate the plot
    plot_metrics(models, args.metric)

if __name__ == "__main__":
    main()