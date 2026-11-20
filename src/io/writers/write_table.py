import csv


def save_metrics_to_csv(history, file_path):
    """
    Save per-epoch metrics to a CSV file.

    `history` is a list of dicts, one per epoch, as returned by `fit`.
    Rows may have different keys; missing values are left blank.
    """
    if not history:
        return

    # Union of all keys, in first-seen order, with "epoch" pinned to the front
    columns = []
    for record in history:
        for key in record:
            if key not in columns:
                columns.append(key)
    if "epoch" in columns:
        columns.remove("epoch")
    columns = ["epoch"] + columns

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, restval="")
        writer.writeheader()

        for index, record in enumerate(history, start=1):
            row = dict(record)
            row.setdefault("epoch", index)
            writer.writerow(row)