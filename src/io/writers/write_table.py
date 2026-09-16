import csv

def save_metrics_to_csv(metrics, file_path):
    """
    Save nested training and validation metrics to a CSV file.

    Each row represents one epoch. Metrics with fewer values are left blank.
    """

    # Flatten the nested dictionary into column names
    columns = []

    for split, split_metrics in metrics.items():
        for metric_name in split_metrics:
            columns.append(f"{split}_{metric_name}")

    # Find the largest number of epochs across all metrics
    number_of_epochs = max(
        (
            len(values)
            for split_metrics in metrics.values()
            for values in split_metrics.values()
        ),
        default=0
    )

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch"] + columns
        )

        writer.writeheader()

        for epoch_index in range(number_of_epochs):
            row = {
                "epoch": epoch_index + 1
            }

            for split, split_metrics in metrics.items():
                for metric_name, values in split_metrics.items():
                    column_name = f"{split}_{metric_name}"

                    if epoch_index < len(values):
                        row[column_name] = values[epoch_index]
                    else:
                        row[column_name] = ""

            writer.writerow(row)