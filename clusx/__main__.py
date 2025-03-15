"""Entry point for direct module execution.

This module serves as the main entry point when the package is executed directly
using ``python -m clusx``. It initializes the command-line interface and passes
control to the main CLI function.
"""

import sys

from clusx.cli import main


def init() -> None:
    """Run clusx.cli.main() when current file is executed by an interpreter.

    If the file is used as a module, the :func:`clusx.cli.main` function will
    not automatically execute. The :func:`sys.exit` function is called with a
    return value of :func:`clusx.cli.main`, as all good UNIX programs do.
    """
    if __name__ == "__main__":
        sys.exit(main())


init()
