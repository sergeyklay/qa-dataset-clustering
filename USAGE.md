# QA Dataset Clustering Toolkit (qadst) - Usage Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Quick Start](#quick-start)
   - [Input Format](#input-format)
3. [Basic Usage](#basic-usage)
   - [Clustering QA Pairs](#clustering-qa-pairs)
   - [Benchmarking Clusters](#benchmarking-clusters)
   - [Understanding Output Files](#understanding-output-files)
4. [Common Workflows](#common-workflows)
   - [End-to-End Example](#end-to-end-example)
   - [Interpreting Results](#interpreting-results)
5. [Configuration Options](#configuration-options)
   - [Command Line Options](#command-line-options)
   - [Environment Variables](#environment-variables)
6. [Advanced Features](#advanced-features)
   - [Embedding Models](#embedding-models)
   - [Embedding Caching](#embedding-caching)
   - [Clustering Parameters](#clustering-parameters)
   - [Noise Point Handling](#noise-point-handling)
   - [Large Cluster Handling](#large-cluster-handling)
7. [Reporting and Visualization](#reporting-and-visualization)
   - [CSV Reports](#csv-reports)
   - [Console Output](#console-output)
   - [Custom Reporters](#custom-reporters)
8. [Topic Labeling](#topic-labeling)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Introduction

The QA Dataset Clustering Toolkit (qadst) helps you organize, analyze, and improve question-answer datasets for Retrieval-Augmented Generation (RAG) systems. It addresses common challenges in RAG evaluation:

- Identifying redundant questions in your dataset
- Separating technical/engineering questions from end-user questions
- Organizing questions into meaningful semantic groups
- Measuring dataset quality quantitatively

This guide will walk you through using qadst, from basic operations to advanced features.

## Getting Started

### Installation

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md).

### Quick Start

Here's a minimal example to get you started:

```bash
# Cluster a QA dataset
qadst cluster --input data/qa_pairs.csv

# Benchmark the clustering results
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv
```

### Input Format

qadst expects a CSV file with at least two columns:
- First column: questions
- Second column: answers

Example:
```csv
question,answer
How do I reset my password?,You can reset your password by clicking on the "Forgot Password" link.
What payment methods do you accept?,We accept credit cards, PayPal, and bank transfers.
```

## Basic Usage

### Clustering QA Pairs

The `cluster` command groups similar questions together:

```bash
qadst cluster --input data/qa_pairs.csv --output-dir ./output --filter
```

This command:
1. Loads QA pairs from the input CSV
2. Deduplicates semantically similar questions
3. Filters out engineering-focused questions (if `--filter` is enabled)
4. Clusters the remaining questions using HDBSCAN
5. Saves the results to the output directory

### Benchmarking Clusters

The `benchmark` command evaluates clustering quality and generates reports:

```bash
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --use-llm
```

This command:
1. Loads the clustering results and original QA pairs
2. Calculates quality metrics for each cluster and globally
3. Generates topic labels for each cluster (using LLM if `--use-llm` is enabled)
4. Creates reports in the output directory

> **Important**: Always use the same embedding model for benchmarking that you used for clustering.
> For example, if you clustered with SBERT, use the same SBERT model for benchmarking:
> ```bash
> # Clustering with SBERT
> qadst cluster --input data/qa_pairs.csv --embedding-model sbert
>
> # Benchmarking with the SAME model
> qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --embedding-model sbert
> ```
> Using different embedding models between clustering and benchmarking will lead to inconsistent results.

### Understanding Output Files

The toolkit generates several output files:

**Clustering Output:**
- `qa_clusters.json`: Clustering results in JSON format
- `qa_cleaned.csv`: Deduplicated and filtered QA pairs
- `engineering_questions.csv`: Questions identified as engineering-focused (if filtering is enabled)

**Benchmarking Output:**
- `cluster_quality_report.csv`: Detailed metrics for each cluster
- Enhanced `qa_clusters.json` with additional metrics and topic labels

## Common Workflows

### End-to-End Example

A typical workflow involves these steps:

1. **Prepare your QA dataset** in CSV format
2. **Run clustering** to group similar questions
   ```bash
   qadst cluster --input data/qa_pairs.csv
   ```
3. **Run benchmarking** to evaluate cluster quality
   ```bash
   qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --use-llm
   ```
4. **Analyze the results** in the output directory
5. **Refine parameters** if needed and re-run clustering
   ```bash
   qadst cluster --input data/qa_pairs.csv --min-cluster-size 50 --min-samples 3
   ```

### Interpreting Results

The benchmarking process generates metrics to help you evaluate clustering quality:

| Metric                    | Good Value   | Interpretation                                           |
| ------------------------- | ------------ | -------------------------------------------------------- |
| Davies-Bouldin Index      | <1.0         | Lower values indicate better cluster separation          |
| Calinski-Harabasz Score   | >100         | Higher values indicate better cluster definition         |
| Silhouette Score          | >0.5         | Higher values indicate better cluster coherence          |
| Coherence Score           | >0.4         | Higher values indicate more semantically similar questions|
| Noise Ratio               | <0.1         | Lower values indicate fewer outliers                     |

**Example:** Poor Quality Clustering

| Metric                    | Threshold   | Current Value                 | Interpretation                                           |
| ------------------------- | ----------- | ----------------------------- | -------------------------------------------------------- |
| Davies-Bouldin Index      | <1.0 ideal  | 4.38                          | Poor cluster separation (clusters overlap significantly) |
| Calinski-Harabasz Score   | >100 good   | 54.66                         | Weak cluster density (clusters are not compact)          |
| Low Coherence Clusters    | >0.4 target | 13,18,20,25,26,29 (0.14-0.34) | Mixed/irrelevant QA pairs in same cluster                |

Possible improvement: Adjust HDBSCAN parameters or try a different embedding model.

## Configuration Options

### Command Line Options

#### Common Options (Both Commands)

- `--embedding-model`: Embedding model to use (default: text-embedding-3-large)
- `--llm-model`: LLM model to use for filtering and topic labeling (default: gpt-4o)
- `--output-dir`: Directory to save output files (default: ./output)

#### Clustering Options

- `--filter/--no-filter`: Enable/disable filtering of engineering questions
- `--min-cluster-size`: Minimum size of clusters (if not provided, calculated automatically)
- `--min-samples`: HDBSCAN min_samples parameter (default: 5)
- `--cluster-selection-epsilon`: HDBSCAN cluster_selection_epsilon parameter (default: 0.3)
- `--cluster-selection-method`: HDBSCAN cluster selection method (default: eom, alternative: leaf)
- `--keep-noise/--cluster-noise`: Keep noise points unclustered or force them into clusters (default: --cluster-noise)

#### Benchmarking Options

- `--use-llm/--no-llm`: Enable/disable LLM for topic labeling
- `--reporters`: Comma-separated list of reporters to enable (default: csv,console)

### Environment Variables

Configure default settings in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## Advanced Features

### Embedding Models

#### Available Models

By default, qadst uses OpenAI's `text-embedding-3-large` model. You can specify a different model:

```bash
# Use a smaller, faster model
qadst cluster --input data/qa_pairs.csv --embedding-model text-embedding-3-small
```

The toolkit supports any embedding model available through the OpenAI API.

#### Using Sentence-BERT (SBERT) Models

qadst also supports local Sentence-BERT models, which don't require an API key and run entirely on your machine:

```bash
# Use the default SBERT model (all-MiniLM-L6-v2)
qadst cluster --input data/qa_pairs.csv --embedding-model sbert

# Use a specific SBERT model
qadst cluster --input data/qa_pairs.csv --embedding-model "sbert:all-mpnet-base-v2"
```

Benefits of SBERT models:
- **No API key required**: Run entirely locally
- **Lower dimensionality**: SBERT models like all-MiniLM-L6-v2 produce 384-dimensional vectors (vs. 3072 for OpenAI)
- **Faster processing**: Local execution can be faster for smaller datasets
- **Cost-effective**: No usage charges

To use SBERT models, you need to install the optional dependency:
```bash
poetry add sentence-transformers
```

> **Note**: When using SBERT for clustering, make sure to use the same SBERT model for benchmarking:
> ```bash
> # Benchmarking with the same SBERT model
> qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --embedding-model sbert
> ```

#### Model Selection Considerations

- **Quality vs. Speed**: Larger models provide better semantic understanding but may be slower and more expensive
- **Dimensionality**: Different models produce embeddings with different dimensions
- **Domain Specificity**: Some models may perform better for specific domains or languages
- **Local vs. API**: SBERT models run locally but may have lower quality than OpenAI models

### Embedding Caching

qadst automatically caches embeddings to improve performance and reduce API costs:

#### How Caching Works

1. **Memory Cache**: Embeddings are stored in memory during a session
2. **Disk Cache**: Embeddings are saved to disk in the output directory
3. **Deterministic Hashing**: A hash of the input questions is used as a cache key
4. **Automatic Invalidation**: Cache is invalidated when the dataset changes

Cache files follow this naming pattern:
```
embeddings_{model_name}_{hash}.npy
```

To clear the cache:
```bash
rm output/embeddings_*.npy
```

### Clustering Parameters

qadst uses HDBSCAN with parameters tuned based on academic research:

- **Logarithmic Scaling**: `min_cluster_size` scales logarithmically with dataset size
- **Optimal Parameters**: Default values are based on clustering literature
- **Cluster Selection Method**: Controls how HDBSCAN selects clusters:
  - `eom` (default): Selects clusters based on stability, resulting in varying cluster sizes
  - `leaf`: Selects leaf nodes, resulting in more fine-grained clusters

Example customization:
```bash
# More fine-grained clusters
qadst cluster --input data/qa_pairs.csv --min-cluster-size 50 --min-samples 3 --cluster-selection-epsilon 0.2
```

### Noise Point Handling

HDBSCAN identifies outliers as "noise points." You can control how these are handled:

- **--cluster-noise** (default): Force noise points into clusters using K-means
  - Ensures all questions are assigned to a cluster
  - May reduce overall cluster quality

- **--keep-noise**: Keep noise points unclustered
  - Preserves HDBSCAN's density-based clustering decisions
  - Improves cluster coherence and separation metrics

Example:
```bash
qadst cluster --input data/qa_pairs.csv --keep-noise
```

### Large Cluster Handling

qadst handles large clusters using a sophisticated approach:

- **Recursive HDBSCAN**: Large clusters are processed using recursive HDBSCAN with stricter parameters
- **Adaptive Fallback**: Falls back to K-means only when necessary
- **Automatic Sizing**: Parameters are adjusted based on cluster size

This ensures that the natural density structure of the data is preserved whenever possible.

## Reporting and Visualization

### CSV Reports

The CSV reporter generates a file named `cluster_quality_report.csv` with:

- **Cluster_ID**: Unique identifier for each cluster
- **Num_QA_Pairs**: Number of question-answer pairs in the cluster
- **Avg_Similarity**: Average pairwise similarity between questions
- **Coherence_Score**: Semantic coherence score (higher is better)
- **Topic_Label**: Descriptive label for the cluster content

### Console Output

The console reporter displays a formatted table with:

1. **Summary Section**: Dataset statistics and global metrics
2. **Top Clusters Table**: The largest clusters with their metrics

Example output:
```
--------------------------------------------------------------------------------
|                           CLUSTER ANALYSIS SUMMARY                           |
--------------------------------------------------------------------------------
Total QA pairs: 5175
Clusters JSON: output/qa_clusters.json

Global Metrics:
  Noise Ratio: 0.00
  Davies-Bouldin Index: 3.95
  Calinski-Harabasz Index: 35.91
  Silhouette Score: 0.12

--------------------------------------------------------------------------------
|                            TOP 5 CLUSTERS BY SIZE                            |
--------------------------------------------------------------------------------
| Cluster ID | Size         | Coherence    | Topic                             |
--------------------------------------------------------------------------------
| 1          | 582          | 0.37         | API Endpoint Behavior             |
| 2          | 537          | 0.49         | Digital Document Workflows        |
| 17         | 206          | 0.14         | Customer Trust Insights           |
| 6          | 168          | 0.43         | API Integration Guidelines        |
| 41         | 158          | 0.52         | Feature-Specific Integrations     |
--------------------------------------------------------------------------------
```

### Custom Reporters

You can select which reporters to use:

```bash
# Use only the CSV reporter
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --reporters csv
```

Developers can create custom reporters by:
1. Implementing a class that inherits from `BaseReporter`
2. Implementing the `generate_report` method
3. Registering the reporter with the `ReporterRegistry`

## Topic Labeling

qadst generates descriptive topic labels for each cluster using:

1. **Context-Aware Generation**: Labels are generated in order of cluster size, with each new label being aware of previously generated labels
2. **Semantic Distinctiveness**: The LLM is instructed to create labels that are semantically distinct
3. **Disambiguation**: When semantic overlap is detected, cluster identifiers are added
4. **Fallback Mechanisms**: If LLM-based labeling fails, the system falls back to TF-IDF/NMF-based keyword extraction

This approach helps address the common challenge of semantic overlap in topic labeling.

## Troubleshooting

### Common Issues

- **Missing API Key**: Ensure `OPENAI_API_KEY` is set in your environment or `.env` file
- **Out of Memory**: For large datasets, try processing in batches or use a machine with more RAM
- **Poor Clustering Quality**: Try adjusting HDBSCAN parameters or using a different embedding model

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the GitHub repository for known issues
2. Open a new issue with a detailed description of your problem

## API Reference

For developers who want to use qadst programmatically, refer to the class documentation:

- `BaseClusterer`: Base class for clustering implementations
- `HDBSCANQAClusterer`: HDBSCAN-based clustering implementation
- `ClusterBenchmarker`: Evaluates and analyzes clustering results
- `BaseReporter`, `CSVReporter`, `ConsoleReporter`: Reporting system

Example programmatic usage:

```python
from qadst import HDBSCANQAClusterer, ClusterBenchmarker

# Choose your embedding model - use the same for both clustering and benchmarking
embedding_model = "sbert"  # or "text-embedding-3-large", etc.

# Create a clusterer with the chosen model
clusterer = HDBSCANQAClusterer(
    embedding_model_name=embedding_model,
    output_dir="./output"
)

# Process a dataset
results = clusterer.process_dataset("data/qa_pairs.csv")

# Create a benchmarker with the SAME model
benchmarker = ClusterBenchmarker(
    embedding_model_name=embedding_model,  # Use the same model as for clustering
    output_dir="./output"
)

# Generate the report
report = benchmarker.generate_cluster_report(
    clusters_json_path="output/qa_clusters.json",
    qa_csv_path="data/qa_pairs.csv",
    use_llm_for_topics=True
)
```
