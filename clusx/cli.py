"""Command-line interface for the Clusterium.

This module provides a command-line interface for clustering text data,
benchmarking clustering results, and generating reports. It handles command-line
arguments, environment configuration, and execution of the appropriate toolkit
functionality based on user commands.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional

from .errors import ClusterIntegrityError, EvaluationError
from .logging import get_logger, setup_logging
from .version import __copyright__, __version__

logger = get_logger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

BANNER = r"""


              ████
             ░░███
      ██████  ░███  █████ ████  █████  █████ █████
     ███░░███ ░███ ░░███ ░███  ███░░  ░░███ ░░███
    ░███ ░░░  ░███  ░███ ░███ ░░█████  ░░░█████░
    ░███  ███ ░███  ░███ ░███  ░░░░███  ███░░░███
    ░░██████  █████ ░░████████ ██████  █████ █████
     ░░░░░░  ░░░░░   ░░░░░░░░ ░░░░░░  ░░░░░ ░░░░░



"""


class RichGroup(click.Group):
    """Custom Click group that displays a banner before the help text."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_help(self, ctx, formatter):
        """Writes the help into the formatter if it exists.

        This method is called by Click when the help text is requested.
        """
        click.secho(BANNER, nl=False)
        super().format_help(ctx, formatter)


def common_options(func: Callable) -> Callable:
    """Common options for all clusx CLI commands."""
    func = click.option(
        "--output-dir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=str(OUTPUT_DIR),
        help="Directory to save output files (default: ./output)",
    )(func)
    return func


@click.group(
    help="Text Clustering Toolkit for Bayesian Nonparametric Analysis",
    cls=RichGroup,
)
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


@cli.command(help="Cluster text data using various Bayesian nonparametric methods")
@common_options
@click.option(
    "--input",
    "input_",
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
    "--dp-alpha",
    default=0.5,
    show_default=True,
    type=float,
    help=(
        "Concentration parameter for Dirichlet Process (α > 0, "
        "typical values 0.1-10). Note: DP only uses α."
    ),
)
@click.option(
    "--dp-kappa",
    default=0.3,
    show_default=True,
    type=float,
    help="Kappa parameter for Dirichlet likelihood model",
)
@click.option(
    "--pyp-alpha",
    default=0.3,
    show_default=True,
    type=float,
    help=(
        "Concentration parameter for Pitman-Yor Process (α > -σ). "
        "Using same α value as DP leads to dramatically clustering behaviors."
    ),
)
@click.option(
    "--pyp-kappa",
    default=0.3,
    show_default=True,
    type=float,
    help="Kappa parameter for Pitman-Yor likelihood model",
)
@click.option(
    "--pyp-sigma",
    default=0.3,
    show_default=True,
    type=float,
    help=(
        "Discount parameter for Pitman-Yor Process (0.0 ≤ σ < 1.0). "
        "PYP uses both α and σ parameters."
    ),
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
    input_: str,
    output: str,
    output_dir: str,
    dp_alpha: float,
    dp_kappa: float,
    pyp_alpha: float,
    pyp_kappa: float,
    pyp_sigma: float,
    random_seed: Optional[int],
    column: Optional[str],
):
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
        logger.debug(
            "Loading data from %s%s...",
            input_,
            ", using column '" + column + "' " if column else "",
        )

        texts = load_data(input_, column)
        _validate_dataset(texts)
        logger.info("Loaded %d texts for clustering", len(texts))

        logger.info("Performing Dirichlet Process clustering...")
        dp = DirichletProcess(
            alpha=dp_alpha,
            kappa=dp_kappa,
            random_state=random_seed,
        )

        clusters_dp = dp.fit_predict(texts)
        logger.info(
            "DP clustering complete. Found %d clusters", len(set(dp.cluster_params))
        )

        logger.info("Performing Pitman-Yor Process clustering...")
        pyp = PitmanYorProcess(
            alpha=pyp_alpha,
            kappa=pyp_kappa,
            sigma=pyp_sigma,
            random_state=random_seed,
        )
        clusters_pyp = pyp.fit_predict(texts)
        logger.info(
            "PYP clustering complete. Found %d clusters", len(set(clusters_pyp))
        )

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
            alpha=dp_alpha,
            sigma=0.0,
            kappa=dp_kappa,
        )
        save_clusters_to_csv(
            pyp_output,
            texts,
            clusters_pyp,
            "PYP",
            alpha=pyp_alpha,
            sigma=pyp_sigma,
            kappa=pyp_kappa,
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
            alpha=dp_alpha,
            sigma=0.0,
            kappa=dp_kappa,
        )
        save_clusters_to_json(
            pyp_json,
            texts,
            clusters_pyp,
            "PYP",
            alpha=pyp_alpha,
            sigma=pyp_sigma,
            kappa=pyp_kappa,
        )
    except Exception as err:  # pylint: disable=broad-except
        logger.exception(err)  # Unexpected error
        sys.exit(1)


@cli.command(help="Evaluate and compare clustering results")
@common_options
@click.option(
    "--input",
    "input_",
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
    "--show-plot/--no-show-plot",
    default=False,
    show_default=True,
    help="Display plots interactively (not recommended for automated runs)",
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
    input_: str,
    dp_clusters: str,
    pyp_clusters: str,
    output_dir: str,
    plot: bool,
    show_plot: bool,
    random_seed: Optional[int],
    column: Optional[str],
):
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
        logger.debug(
            "Loading data from %s%s...",
            input_,
            ", using column '" + column + "' " if column else "",
        )

        texts = load_data(input_, column)
        _validate_dataset(texts)

        logger.info("Loaded %d texts for evaluation", len(texts))

        # Load cluster assignments
        logger.debug("Loading DP cluster assignments from %s...", dp_clusters)
        dp_cluster_assignments, dp_params = load_cluster_assignments(dp_clusters)

        logger.debug("Loading PYP cluster assignments from %s...", pyp_clusters)
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
            kappa=dp_params["kappa"],
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
            kappa=pyp_params["kappa"],
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
            visualize_evaluation_dashboard(reports, output_dir, show_plot=show_plot)
            if show_plot:
                logger.info("Close the plot window to continue.")

        logger.info("Evaluation complete.")
    except (ClusterIntegrityError, EvaluationError) as error:
        logger.error(error)
        sys.exit(1)
    except Exception as error:  # pylint: disable=broad-except
        logger.exception(error)  # Unexpected error
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
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Handle unexpected errors
        logger.error(exc)
        return 1
