# Usage Guide

This document provides detailed instructions for using the QA Dataset Clustering Tool (qadst).

## Command Line Interface

The `qadst` command-line tool provides a simple interface for clustering question-answer datasets.

### Basic Usage

```bash
qadst --input your_data.csv --output clusters.csv
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input CSV file (required) | - |
| `--column` | Column name to use for clustering | "question" |
| `--output` | Output CSV file path | "clusters_output.csv" |
| `--output-dir` | Directory to save output files | "output" |
| `--alpha` | Concentration parameter for clustering | 1.0 |
| `--sigma` | Discount parameter for Pitman-Yor Process | 0.5 |
| `--plot` | Generate cluster distribution plot | False |
| `--cache-dir` | Directory to cache embeddings | ".cache" |

### Examples

#### Specifying Column Names

If your CSV file has a different column name for questions:

```bash
qadst --input your_data.csv --column question_text --output clusters.csv
```

#### Adjusting Clustering Parameters

Fine-tune the clustering by adjusting the alpha and sigma parameters:

```bash
qadst --input your_data.csv --alpha 0.5 --sigma 0.3
```

#### Generating Visualizations

Generate plots showing the distribution of cluster sizes:

```bash
qadst --input your_data.csv --plot
```

#### Specifying Output Directory

Save all output files to a specific directory:

```bash
qadst --input your_data.csv --output-dir results
```

## Python API

You can also use the clustering functionality directly in your Python code.

### Basic Usage

```python
from qadst.clustering import DirichletProcess, PitmanYorProcess, EmbeddingCache
from qadst.clustering.utils import load_data_from_csv, save_clusters_to_json

# Load data
texts, data = load_data_from_csv("your_data.csv", column="question")

# Create cache provider
cache = EmbeddingCache(cache_dir=".cache")

# Perform Dirichlet Process clustering
dp = DirichletProcess(alpha=1.0, cache=cache)
clusters, params = dp.fit(texts)

# Save results
save_clusters_to_json("clusters.json", texts, clusters, "DP", data)
```

### Using Pitman-Yor Process

The Pitman-Yor Process often produces better clustering results for text data:

```python
# Perform Pitman-Yor Process clustering
pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=cache)
clusters_pyp, params_pyp = pyp.fit(texts)

# Save results
save_clusters_to_json("pyp_clusters.json", texts, clusters_pyp, "PYP", data)
```

### Customizing the Clustering Process

You can customize various aspects of the clustering process:

```python
# Custom alpha and sigma values
dp = DirichletProcess(alpha=0.5, cache=cache)
pyp = PitmanYorProcess(alpha=0.5, sigma=0.3, cache=cache)

# Custom embedding model (advanced)
from sentence_transformers import SentenceTransformer
custom_model = SentenceTransformer("all-mpnet-base-v2")  # Different model

# Custom similarity function (advanced)
def custom_similarity(text, cluster_param):
    # Your custom similarity logic here
    pass
```

## Output Files

The tool generates several output files:

- `*_dp.csv`: CSV file with Dirichlet Process clustering results
- `*_pyp.csv`: CSV file with Pitman-Yor Process clustering results
- `*_dp.json`: JSON file with Dirichlet Process clustering results
- `*_pyp.json`: JSON file with Pitman-Yor Process clustering results
- `qa_clusters.json`: Combined JSON file with clustering results
- `*_clusters.png`: Visualization of cluster size distributions (if `--plot` is specified)

### JSON Output Format

The JSON output follows this structure:

```json
{
  "clusters": [
    {
      "id": 1,
      "representative": [
        {
          "question": "What is the capital of France?",
          "answer": "Paris is the capital of France."
        }
      ],
      "source": [
        {
          "question": "What is the capital of France?",
          "answer": "Paris is the capital of France."
        },
        {
          "question": "What city is the capital of France?",
          "answer": "Paris is the capital city of France."
        }
      ]
    }
  ]
}
```

Each cluster has:
- A unique ID
- A representative question-answer pair (typically the first item in the cluster)
- A list of source question-answer pairs that belong to the cluster

## Performance Considerations

- **Caching**: Embeddings are cached to speed up repeated runs. Use the `--cache-dir` option to specify a cache directory.
- **Memory Usage**: Large datasets may require significant memory, especially for the embedding model.
- **Processing Time**: The clustering process can be time-consuming for large datasets. The Pitman-Yor Process is typically faster than the Dirichlet Process.

## Troubleshooting

If you encounter issues:

1. Check your input CSV file format
2. Ensure you have sufficient memory for large datasets
3. Try adjusting the alpha and sigma parameters for better clustering results

For more help, please open an issue on the [GitHub repository](https://github.com/sergeyklay/qa-dataset-clustering/issues).
