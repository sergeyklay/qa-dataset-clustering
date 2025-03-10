# Installing qadst

This document provides detailed instructions for installing the QA Dataset Clustering Toolkit (qadst).

## Prerequisites

Before installing qadst, ensure you have the following prerequisites:

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- OpenAI API key (for embedding models and LLM-based features)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/sergeyklay/qa-dataset-clustering.git
cd qa-dataset-clustering
```

### 2. Install Dependencies

Using Poetry:

```bash
poetry install
```

This will create a virtual environment and install all required dependencies.

#### Optional Dependencies

For local embedding models (SBERT):

```bash
poetry add sentence-transformers
```

This allows you to use Sentence-BERT models locally without requiring an OpenAI API key for embeddings.

### 3. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit the `.env` file to include your API keys and other configuration:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 4. Verify Installation

To verify that qadst is installed correctly, run:

```bash
poetry run qadst --help
```

You should see the help message with available commands and options.

## Development Installation

If you're planning to contribute to qadst, install with development dependencies:

```bash
poetry install --with dev
```

This will install additional tools for testing, linting, and documentation.

## Troubleshooting Installation

### Common Issues

- **Poetry not found**: Ensure Poetry is installed and in your PATH
- **Python version mismatch**: Verify you have Python 3.10+ installed
- **Dependency conflicts**: Try `poetry update` to resolve dependency issues
- **Missing sentence-transformers**: If you want to use SBERT embeddings, make sure to install the optional dependency with `poetry add sentence-transformers`

If you encounter any other installation issues, please check the GitHub repository for known issues or open a new issue with details about your environment and the error messages you're seeing.
