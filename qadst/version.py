"""Version information.

This module provides version and metadata information for the package.
It attempts to get this information from the installed package metadata,
with fallback values for development environments.
"""

import importlib.metadata

try:
    # Get package metadata from Poetry
    _package_metadata = importlib.metadata.metadata("qadst")

    __version__ = _package_metadata.get("Version", "0.0.0+dev")
    __description__ = _package_metadata.get("Summary", "QA Dataset Clustering Toolkit")
    __license__ = _package_metadata.get("License", "MIT")
    __author__ = _package_metadata.get("Author", "Serghei Iakovlev")
    __author_email__ = _package_metadata.get("Author-email", "oss@serghei.pl")
    __url__ = _package_metadata.get(
        "Home-page", "https://github.com/sergeyklay/qa-dataset-clustering"
    )

except importlib.metadata.PackageNotFoundError:
    # Package is not installed, use fallback values for development
    __version__ = "0.0.0+dev"
    __description__ = "QA Dataset Clustering Toolkit"
    __license__ = "MIT"
    __author__ = "Serghei Iakovlev"
    __author_email__ = "oss@serghei.pl"
    __url__ = "https://github.com/sergeyklay/qa-dataset-clustering"

# Derived values
__copyright__ = f"Copyright (C) 2025 {__author__}"
