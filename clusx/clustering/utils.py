"""
Utility functions for data loading, saving, and visualization.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from clusx.errors import MissingClusterColumnError, MissingParametersError
from clusx.logging import get_logger
from clusx.utils import to_numpy

if TYPE_CHECKING:
    from typing import Optional, Union

    import torch
    from numpy.typing import NDArray

    EmbeddingTensor = Union[torch.Tensor, NDArray[np.float32]]


logger = get_logger(__name__)


def is_csv_file(input_file: str) -> bool:
    """
    Determine if a file is a CSV file based on extension and content.

    Args:
        input_file: Path to the input file

    Returns:
        bool: True if the file is likely a CSV, False otherwise
    """
    # First check file extension
    if input_file.lower().endswith(".csv"):
        return True

    # For files without .csv extension, try to detect CSV format
    is_csv = False
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            # Read a sample of the file to determine if it's CSV
            sample = f.read(4096)  # Read a reasonable sample size
            if sample:  # Check if we got any content
                # Try to detect CSV with Sniffer
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    if dialect.delimiter in [",", ";", "\t"]:
                        is_csv = True
                except csv.Error:
                    # Not a CSV according to the sniffer
                    pass
    except OSError:
        # Handle file access errors
        logger.warning("Error accessing file %s", input_file)

    return is_csv


def load_data(input_file: str, column: Optional[str] = None) -> list[str]:
    """
    Load text data from a file. Supports text files and CSV files.

    Args:
        input_file: Path to the input file (text or CSV)
        column: Column name containing the text data (required for CSV files)

    Returns:
        list[str]: A list of texts

    Raises:
        ValueError: If a CSV file is provided without specifying a column
    """
    texts = []
    if is_csv_file(input_file):
        if column is None:
            raise ValueError("Column name must be specified when using a CSV file")

        df = pd.read_csv(input_file)
        if column in df.columns:
            texts = df[column].tolist()
        else:
            logger.warning(
                "Column '%s' not found in CSV. Available columns: %s",
                column,
                ", ".join(df.columns),
            )
            raise ValueError(f"Column '{column}' not found in the CSV file")
    else:
        # Process as a text file (one text per line)
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    texts.append(line)

    return texts


def save_clusters_to_csv(
    output_file: str,
    texts: list[str],
    clusters: list[int],
    model_name: str,
    alpha: float,
    sigma: float,
    kappa: float,
) -> None:
    """
    Save clustering results to a CSV file.

    Args:
        output_file: Path to the output CSV file
        texts: List of text strings
        clusters: List of cluster assignments
        model_name: Name of the clustering model
        alpha: Concentration parameter
        sigma: Discount parameter
        kappa: Kappa parameter for likelihood model
    """
    df = pd.DataFrame(
        {
            "Text": texts,
            f"Cluster_{model_name}": clusters,
            "Alpha": [alpha] * len(texts),
            "Sigma": [sigma] * len(texts),
            "Kappa": [kappa] * len(texts),
        }
    )

    df.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    logger.debug("Clustering results saved to %s", output_file)


def save_clusters_to_json(
    output_file: str,
    texts: list[str],
    clusters: list[int],
    model_name: str,
    alpha: float,
    sigma: float,
    kappa: float,
) -> None:
    """
    Save clustering results to a JSON file.

    Args:
        output_file: Path to the output JSON file
        texts: List of text strings
        clusters: List of cluster assignments
        model_name: Name of the clustering model
        alpha: Concentration parameter
        sigma: Discount parameter
        kappa: Kappa parameter for likelihood model
    """
    # Group texts by cluster
    cluster_groups = defaultdict(list)
    for text, cluster_id in zip(texts or [], clusters or []):
        cluster_groups[cluster_id].append(text)

    clusters_json = {
        "clusters": [],
        "metadata": {
            "model_name": model_name,
            "alpha": alpha,
            "sigma": sigma,
            "kappa": kappa,
        },
    }

    for i, (cluster_id, cluster_texts) in enumerate(cluster_groups.items()):
        representative_text = cluster_texts[0]

        # Create the cluster object with the new format
        cluster_obj = {
            "id": i + 1,
            "representative": representative_text,
            "members": cluster_texts,
        }

        clusters_json["clusters"].append(cluster_obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)
    logger.debug("JSON clusters saved to %s", output_file)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        Numpy array of embeddings
    """
    from datetime import datetime

    from clusx.clustering import DirichletProcess

    # TODO: Extract embedding generation to a separate function/class
    # Use default parameters for embedding generation only
    dp = DirichletProcess(alpha=1.0, kappa=1.0)
    embeddings = []

    # Process texts with progress bar
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for text in tqdm(
        texts,
        desc=f"{timestamp} - INFO - Computing embeddings",
        total=len(texts),
        disable=None,  # Disable on non-TTY
        unit=" texts",
    ):
        emb_array = to_numpy(dp.get_embedding(text))
        embeddings.append(emb_array)

    return np.array(embeddings)


def load_cluster_assignments(csv_path: str) -> tuple[list[int], dict[str, float]]:
    """
    Load cluster assignments and parameters from a CSV file.

    Args:
        csv_path: Path to the CSV file containing cluster assignments

    Returns:
        tuple[list[int], dict[str, float]]: A tuple containing:
            - List of cluster assignments (clustered texts)
            - Dictionary of parameters (alpha, sigma, kappa)

    Raises:
        MissingClusterColumnError: If no cluster column is found in the file
        MissingParametersError: If required parameters are missing in the file
    """
    df = pd.read_csv(csv_path)

    # Check which column contains the cluster assignments
    cluster_column = None
    for col in df.columns:
        # Cluster_PYP or Cluster_DP
        if col.lower().startswith("cluster_"):
            cluster_column = col
            break

    if not cluster_column:
        raise MissingClusterColumnError(csv_path)

    # Extract cluster assignments
    cluster_assignments = df[cluster_column].tolist()
    params = {}

    # Check if parameter columns exist in the CSV
    if "Alpha" in df.columns:
        params["alpha"] = float(df["Alpha"].iloc[0])

    if "Sigma" in df.columns:
        params["sigma"] = float(df["Sigma"].iloc[0])

    if "Kappa" in df.columns:
        params["kappa"] = float(df["Kappa"].iloc[0])

    missing_params = [
        key
        for key in ["alpha", "sigma", "kappa"]
        if key not in params or params[key] is None
    ]
    if missing_params:
        raise MissingParametersError(csv_path, missing_params)

    return cluster_assignments, params


def load_parameters_from_json(json_path: str) -> dict[str, float]:
    """
    Load clustering parameters from a JSON file.

    Args:
        json_path: Path to the JSON file containing clustering results

    Returns:
        dict[str, float]: A dictionary of parameters (alpha, sigma, kappa)
    """
    # TODO: Do I really need defaults?
    params = {"alpha": 1.0, "sigma": 0.0, "kappa": 1.0}  # Default values

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # TODO: Should I throw an error if metadata or its keys are missing?
        # Check if metadata is available in the JSON
        if "metadata" in data:
            if "alpha" in data["metadata"]:
                params["alpha"] = float(data["metadata"]["alpha"])
            if "sigma" in data["metadata"]:
                params["sigma"] = float(data["metadata"]["sigma"])
            if "kappa" in data["metadata"]:
                params["kappa"] = float(data["metadata"]["kappa"])
    except (OSError, json.JSONDecodeError) as err:
        logger.error("Error loading parameters from JSON: %s", err)

    return params
