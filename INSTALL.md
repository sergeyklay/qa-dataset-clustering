# Installation Guide

This document provides detailed instructions for installing the QA Dataset Clustering Toolkit (qadst).

## Prerequisites

Before installing qadst, ensure you have the following prerequisites:

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- Git (for cloning the repository)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/sergeyklay/qa-dataset-clustering.git
   cd qa-dataset-clustering
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Verifying Installation

To verify that the installation was successful, run:

```bash
qadst --help
```

Or using the Python module:

```bash
python -m qadst --help
```

You should see the help message with available command-line options.

## Dependencies

The main dependencies include:

- `numpy`: For numerical operations
- `sentence-transformers`: For text embeddings
- `scipy`: For distance calculations
- `matplotlib`: For visualization
- `torch`: For deep learning operations
- `tqdm`: For progress bars

These dependencies will be automatically installed by Poetry.

## Troubleshooting

If you encounter any issues during installation:

1. Ensure you have the correct Python version (3.10+)
2. Make sure Poetry is properly installed
3. Check for any error messages during the installation process

For more detailed help, please open an issue on the [GitHub repository](https://github.com/sergeyklay/qa-dataset-clustering/issues).
