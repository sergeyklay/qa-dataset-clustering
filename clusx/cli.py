"""Command-line interface for the Clusterium.

This module provides a command-line interface for clustering text data,
benchmarking clustering results, and generating reports. It handles command-line
arguments, environment configuration, and execution of the appropriate toolkit
functionality based on user commands.

"""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import click

from .logging import get_logger, setup_logging
from .version import __copyright__, __version__

logger = get_logger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


def common_options(func: Callable) -> Callable:
    """Common options for all clusx CLI commands."""
    func = click.option(
        "--output-dir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=str(OUTPUT_DIR),
        help="Directory to save output files (default: ./output)",
    )(func)
    return func


@click.group(help="Text Clustering Toolkit for Bayesian Nonparametric Analysis")
@click.version_option(
    version=__version__,
    prog_name="clusx",
    message=f"""%(prog)s %(version)s
{__copyright__}
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.""",
)
def cli():
    """Text Clustering Toolkit for statistical analysis and benchmarking."""
    pass


@cli.command(help="Cluster text data using various Bayesian nonparametric methods")
@common_options
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the input file (text file or CSV)",
    required=True,
)
@click.option(
    "--output",
    default="clusters_output.csv",
    show_default=True,
    help="CSV file with clustering results",
)
@click.option(
    "--alpha",
    default=5.0,
    show_default=True,
    type=float,
    help="Concentration parameter",
)
@click.option(
    "--sigma",
    default=0.5,
    show_default=True,
    type=float,
    help="Discount parameter for Pitman-Yor",
)
@click.option(
    "--variance",
    default=0.1,
    show_default=True,
    type=float,
    help="Variance parameter for likelihood model",
)
@click.option(
    "--random-seed",
    default=None,
    show_default=True,
    type=int,
    help="Random seed for reproducible clustering",
)
@click.option(
    "--column",
    default=None,
    help="Column name to use for clustering (required for CSV files)",
)
def cluster(
    input: str,
    output: str,
    output_dir: str,
    alpha: float,
    sigma: float,
    variance: float,
    random_seed: Optional[int],
    column: Optional[str],
) -> None:
    """Cluster text data using Dirichlet Process and Pitman-Yor Process."""
    from .clustering import (
        DirichletProcess,
        PitmanYorProcess,
    )
    from .clustering.utils import (
        load_data,
        save_clusters_to_csv,
        save_clusters_to_json,
    )

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"Loading data from {input} "
            f"""{", using column '" + column + "' " if column else ""}..."""
        )

        texts = load_data(input, column)
        _validate_dataset(texts)

        logger.info(f"Loaded {len(texts)} texts for clustering")

        base_measure = {"variance": variance}

        logger.info("Performing Dirichlet Process clustering...")
        dp = DirichletProcess(
            alpha=alpha,
            base_measure=base_measure,
            random_state=random_seed,
        )
        clusters_dp, _ = dp.fit(texts)
        logger.info(f"DP clustering complete. Found {len(set(clusters_dp))} clusters")

        logger.info("Performing Pitman-Yor Process clustering...")
        pyp = PitmanYorProcess(
            alpha=float(alpha),
            sigma=float(sigma),
            base_measure=base_measure,
            random_state=random_seed,
        )
        clusters_pyp, _ = pyp.fit(texts)
        logger.info(f"PYP clustering complete. Found {len(set(clusters_pyp))} clusters")

        # Save results
        output_basename = os.path.basename(output)

        # Save CSV files
        dp_output = os.path.join(output_dir, output_basename.replace(".csv", "_dp.csv"))
        pyp_output = os.path.join(
            output_dir, output_basename.replace(".csv", "_pyp.csv")
        )
        save_clusters_to_csv(
            dp_output,
            texts,
            clusters_dp,
            "DP",
            alpha=alpha,
            sigma=0.0,
            variance=variance,
        )
        save_clusters_to_csv(
            pyp_output,
            texts,
            clusters_pyp,
            "PYP",
            alpha=float(alpha),
            sigma=float(sigma),
            variance=variance,
        )

        # Save JSON files
        dp_json = os.path.join(output_dir, output_basename.replace(".csv", "_dp.json"))
        pyp_json = os.path.join(
            output_dir, output_basename.replace(".csv", "_pyp.json")
        )
        save_clusters_to_json(
            dp_json,
            texts,
            clusters_dp,
            "DP",
            alpha=float(alpha),
            sigma=0.0,
            variance=variance,
        )
        save_clusters_to_json(
            pyp_json,
            texts,
            clusters_pyp,
            "PYP",
            alpha=float(alpha),
            sigma=float(sigma),
            variance=variance,
        )
    except Exception as err:
        logger.error(err)
        sys.exit(1)


@cli.command(help="Evaluate clustering results")
@common_options
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the input file (text file or CSV)",
    required=True,
)
@click.option(
    "--dp-clusters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to Dirichlet Process clustering results CSV",
    required=True,
)
@click.option(
    "--pyp-clusters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to Pitman-Yor Process clustering results CSV",
    required=True,
)
@click.option(
    "--plot/--no-plot",
    default=True,
    show_default=True,
    help="Generate evaluation plots",
)
@click.option(
    "--random-seed",
    default=None,
    show_default=True,
    type=int,
    help="Random seed for reproducible evaluation",
)
@click.option(
    "--column",
    default=None,
    help="Column name to use for clustering (required for CSV files)",
)
def evaluate(
    input: str,
    dp_clusters: str,
    pyp_clusters: str,
    output_dir: str,
    plot: bool,
    random_seed: Optional[int],
    column: Optional[str],
) -> None:
    """Evaluate clustering results using established metrics."""
    from .clustering.utils import (
        get_embeddings,
        load_cluster_assignments,
        load_data,
    )
    from .evaluation import (
        ClusterEvaluator,
        save_evaluation_report,
    )

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"Loading data from {input} "
            f"""{", using column '" + column + "' " if column else ""}..."""
        )

        texts = load_data(input, column)
        _validate_dataset(texts)

        logger.info(f"Loaded {len(texts)} texts for evaluation")

        # Load cluster assignments
        logger.info(f"Loading DP cluster assignments from {dp_clusters}...")
        dp_cluster_assignments, dp_params = load_cluster_assignments(dp_clusters)

        logger.info(f"Loading PYP cluster assignments from {pyp_clusters}...")
        pyp_cluster_assignments, pyp_params = load_cluster_assignments(pyp_clusters)

        embeddings = get_embeddings(texts)

        # Evaluate DP clusters
        logger.info("Evaluating Dirichlet Process clustering...")
        dp_evaluator = ClusterEvaluator(
            texts,
            embeddings,
            dp_cluster_assignments,
            "Dirichlet",
            alpha=dp_params["alpha"],
            sigma=dp_params["sigma"],
            variance=dp_params.get("variance", 0.1),
            random_state=random_seed,
        )
        dp_report = dp_evaluator.generate_report()

        # Evaluate PYP clusters
        logger.info("Evaluating Pitman-Yor Process clustering...")
        pyp_evaluator = ClusterEvaluator(
            texts,
            embeddings,
            pyp_cluster_assignments,
            "Pitman-Yor",
            alpha=pyp_params["alpha"],
            sigma=pyp_params["sigma"],
            variance=pyp_params.get("variance", 0.1),
            random_state=random_seed,
        )
        pyp_report = pyp_evaluator.generate_report()

        reports = {
            "Dirichlet": dp_report,
            "Pitman-Yor": pyp_report,
        }

        save_evaluation_report(reports, output_dir)

        if plot:
            from .visualization import visualize_evaluation_dashboard

            logger.info("Generating evaluation dashboard...")
            try:
                dashboard_path = visualize_evaluation_dashboard(
                    reports, output_dir, show_plot=True
                )
                logger.info(f"Visualization saved to: {dashboard_path}")
                logger.info("Close the plot window to continue.")
            except Exception as e:
                logger.error(f"Error generating visualization: {e}")

        logger.info("Evaluation complete.")
    except Exception as error:
        logger.error(error)
        sys.exit(1)


def _validate_dataset(texts):
    """
    Validates the input dataset for clustering and provides appropriate warnings.

    Checks if the dataset is empty or too small for effective Bayesian nonparametric
    clustering. Displays color-coded warnings based on severity or raises an error
    if the dataset is empty.

    Args:
        texts: List of texts to be clustered
    """
    if not texts:
        raise click.ClickException("No data found in the provided source.")

    if len(texts) < 10:
        click.echo(
            click.style(
                "Warning: Dataset is very small (< 10 texts). "
                "Some evaluation metrics and visualizations may not be available "
                "or meaningful.",
                fg="yellow",
                bold=True,
            )
        )
        return

    if len(texts) <= 2:
        click.echo(
            click.style(
                "Critical: Dataset has only 1-2 texts. "
                "Most evaluation metrics require at least 3 texts. "
                "Consider using a larger dataset for meaningful evaluation.",
                fg="red",
                bold=True,
            )
        )


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the clusx CLI.

    Args:
        args: Command line arguments (uses :py:data:`sys.argv` if None)

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Set up logging
    setup_logging()

    try:
        # Invoke the Click command
        cli.main(args=args, standalone_mode=False)
        return 0
    except click.exceptions.Abort:
        # Handle keyboard interrupts gracefully
        logger.warning("Operation aborted by user")
        return 130  # Standard exit code for SIGINT
    except click.exceptions.Exit as e:
        # Handle normal exit
        return e.exit_code
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error: {e}")
        return 1
