"""
Utility functions for data loading, saving, and visualization.
"""

import csv
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from qadst.logging import get_logger

logger = get_logger(__name__)


def plot_cluster_distribution(
    cluster_assignments: List[int], title: str, color: str
) -> None:
    """
    Plot the distribution of cluster sizes.

    Args:
        cluster_assignments: List of cluster assignments
        title: Title for the plot
        color: Color for the bars
    """
    cluster_counts = Counter(cluster_assignments)
    sizes = sorted(cluster_counts.values(), reverse=True)
    plt.bar(range(1, len(sizes) + 1), sizes, color=color, alpha=0.6)
    plt.xlabel("Cluster Rank")
    plt.ylabel("Cluster Size")
    plt.title(title)


def load_data_from_csv(
    csv_file: str, column: str = "question", answer_column: str = "answer"
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Load text data from a CSV file.

    Args:
        csv_file: Path to the CSV file
        column: Column name containing the text data (default: "question")
        answer_column: Column name containing the answers (default: "answer")

    Returns:
        Tuple containing (texts, data_rows)
    """
    texts = []
    data = []  # Store full data including answers

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column in row and row[column].strip():
                texts.append(row[column])
                # Store the full row data
                data.append(row)

    return texts, data


def save_clusters_to_csv(
    output_file: str, texts: List[str], clusters: List[int], model_name: str
) -> None:
    """
    Save clustering results to a CSV file.

    Args:
        output_file: Path to the output CSV file
        texts: List of text strings
        clusters: List of cluster assignments
        model_name: Name of the clustering model
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", f"Cluster_{model_name}"])
        for text, cluster in zip(texts, clusters):
            writer.writerow([text, cluster])
    logger.info(f"Clustering results saved to {output_file}")


def save_clusters_to_json(
    output_file: str,
    texts: List[str],
    clusters: List[int],
    model_name: str,
    data: Optional[List[Dict[str, Any]]] = None,
    answer_column: str = "answer",
) -> None:
    """
    Save clustering results to a JSON file.

    Args:
        output_file: Path to the output JSON file
        texts: List of text strings
        clusters: List of cluster assignments
        model_name: Name of the clustering model
        data: List of data rows containing answers (optional)
        answer_column: Column name containing the answers (default: "answer")
    """
    cluster_groups = {}
    data_map = {}

    # Create a mapping from question to data row if data is provided
    if data:
        for row in data:
            if "question" in row:
                data_map[row["question"]] = row

    # Group texts by cluster
    for text, cluster_id in zip(texts, clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(text)

    clusters_json = {"clusters": []}

    for i, (cluster_id, cluster_texts) in enumerate(cluster_groups.items()):
        representative_text = cluster_texts[0]

        # Get the answer for the representative text
        representative_answer = "No answer available"
        if (
            data_map
            and representative_text in data_map
            and answer_column in data_map[representative_text]
        ):
            representative_answer = data_map[representative_text][answer_column]
        else:
            representative_answer = f"Answer for cluster {i+1} using {model_name}"

        # Create the cluster object
        cluster_obj = {
            "id": i + 1,
            "representative": [
                {
                    "question": representative_text,
                    "answer": representative_answer,
                }
            ],
            "source": [],
        }

        # Add sources with their real answers if available
        for text in cluster_texts:
            answer = f"Answer for question in cluster {i+1}"
            if data_map and text in data_map and answer_column in data_map[text]:
                answer = data_map[text][answer_column]

            cluster_obj["source"].append({"question": text, "answer": answer})

        clusters_json["clusters"].append(cluster_obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON clusters saved to {output_file}")
