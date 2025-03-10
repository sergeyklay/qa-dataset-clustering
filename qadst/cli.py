"""
Command-line interface for QA Dataset Clustering.
"""

import argparse
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt

from qadst.clustering import (
    DirichletProcess,
    EmbeddingCache,
    PitmanYorProcess,
)
from qadst.clustering.utils import (
    load_data_from_csv,
    plot_cluster_distribution,
    save_clusters_to_csv,
    save_clusters_to_json,
)
from qadst.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Set up logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Text clustering using Dirichlet Process and Pitman-Yor Process"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--column",
        default="question",
        help="Column name to use for clustering (default: question)",
    )
    parser.add_argument(
        "--output", default="clusters_output.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output files (default: output)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Concentration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Discount parameter for Pitman-Yor (default: 0.5)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate cluster distribution plot"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory to cache embeddings (default: .cache)",
    )

    parsed_args = parser.parse_args(args)

    try:
        os.makedirs(parsed_args.cache_dir, exist_ok=True)
        os.makedirs(parsed_args.output_dir, exist_ok=True)

        input_file = parsed_args.input
        column_name = parsed_args.column
        logger.info(f"Loading data from {input_file}, using column '{column_name}'...")

        texts, data = load_data_from_csv(input_file, column_name)
        if not texts:
            logger.error(
                f"No data found in column '{column_name}'. Please check your CSV file."
            )
            return 1
        logger.info(f"Loaded {len(texts)} texts for clustering")

        # Create cache provider
        cache_provider = EmbeddingCache(cache_dir=parsed_args.cache_dir)

        logger.info("Performing Dirichlet Process clustering...")
        dp = DirichletProcess(
            alpha=parsed_args.alpha, base_measure=None, cache=cache_provider
        )
        clusters_dp, params_dp = dp.fit(texts)
        logger.info(f"DP clustering complete. Found {len(set(clusters_dp))} clusters")

        logger.info("Performing Pitman-Yor Process clustering...")
        pyp = PitmanYorProcess(
            alpha=parsed_args.alpha,
            sigma=parsed_args.sigma,
            base_measure=None,
            cache=cache_provider,
        )
        clusters_pyp, params_pyp = pyp.fit(texts)
        logger.info(f"PYP clustering complete. Found {len(set(clusters_pyp))} clusters")

        output_basename = os.path.basename(parsed_args.output)
        dp_output = os.path.join(
            parsed_args.output_dir, output_basename.replace(".csv", "_dp.csv")
        )
        pyp_output = os.path.join(
            parsed_args.output_dir, output_basename.replace(".csv", "_pyp.csv")
        )
        save_clusters_to_csv(dp_output, texts, clusters_dp, "DP")
        save_clusters_to_csv(pyp_output, texts, clusters_pyp, "PYP")

        dp_json = os.path.join(
            parsed_args.output_dir, output_basename.replace(".csv", "_dp.json")
        )
        pyp_json = os.path.join(
            parsed_args.output_dir, output_basename.replace(".csv", "_pyp.json")
        )
        save_clusters_to_json(dp_json, texts, clusters_dp, "DP", data)
        save_clusters_to_json(pyp_json, texts, clusters_pyp, "PYP", data)

        qa_clusters_path = os.path.join(parsed_args.output_dir, "qa_clusters.json")
        save_clusters_to_json(qa_clusters_path, texts, clusters_dp, "Combined", data)
        logger.info(f"Combined clusters saved to {qa_clusters_path}")

        if parsed_args.plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plot_cluster_distribution(
                clusters_dp, "Dirichlet Process Cluster Sizes", "blue"
            )
            plt.subplot(1, 2, 2)
            plot_cluster_distribution(
                clusters_pyp, "Pitman-Yor Process Cluster Sizes", "red"
            )
            plt.tight_layout()
            plot_path = os.path.join(
                parsed_args.output_dir, output_basename.replace(".csv", "_clusters.png")
            )
            plt.savefig(plot_path)
            logger.info(f"Cluster distribution plot saved to {plot_path}")
            plt.show()

        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
