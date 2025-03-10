#!/bin/bash

# Matrix test script for qadst clustering parameters
# This script runs qadst cluster with different combinations of parameters,
# followed by benchmarking each result.
# It reuses the existing embedding cache to avoid redundant API calls.

# Input file (required)
INPUT_FILE="input.csv"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    echo "Please provide a valid input CSV file."
    exit 1
fi

# Use the standard output directory to reuse caches
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# Ensure the embedding cache directory exists
CACHE_DIR="$OUTPUT_DIR/embedding_cache"
if [ ! -d "$CACHE_DIR" ]; then
    echo "Creating embedding cache directory: $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
else
    echo "Using existing embedding cache: $CACHE_DIR"
    echo "This will significantly speed up the matrix test by reusing embeddings."
fi

# Backup original output files if they exist
BACKUP_DIR="$OUTPUT_DIR/backup_matrix_test"
mkdir -p "$BACKUP_DIR"
echo "Backing up original output files to $BACKUP_DIR"
cp "$OUTPUT_DIR/qa_clusters.json" "$BACKUP_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR/qa_cleaned.csv" "$BACKUP_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR/engineering_questions.csv" "$BACKUP_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR/cluster_quality_report.csv" "$BACKUP_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR/filter_cache.json" "$BACKUP_DIR/" 2>/dev/null || true

# Create a directory for the test results
RESULTS_DIR="$OUTPUT_DIR/matrix_tests"
mkdir -p "$RESULTS_DIR"

# Create a markdown report file in the output directory
REPORT_FILE="$OUTPUT_DIR/matrix_test_report.md"
echo "# QADST Parameter Matrix Test Results" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "Input file: $INPUT_FILE" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Define parameter values for the matrix test
MIN_SAMPLES_VALUES=(5 10 15)
EPSILON_VALUES=(0.1 0.2 0.3)
NOISE_OPTIONS=("--keep-noise" "--cluster-noise")

# Counter for run number
RUN_NUMBER=1

# Run the matrix test
for min_samples in "${MIN_SAMPLES_VALUES[@]}"; do
    for epsilon in "${EPSILON_VALUES[@]}"; do
        for noise_option in "${NOISE_OPTIONS[@]}"; do
            echo "========================================================"
            echo "Run $RUN_NUMBER: --min-samples $min_samples --cluster-selection-epsilon $epsilon $noise_option"
            echo "========================================================"

            # Create a directory for this run
            RUN_DIR="$RESULTS_DIR/run_$RUN_NUMBER"
            mkdir -p "$RUN_DIR"

            # Run qadst cluster with the current parameter combination
            # Use the main output directory to reuse caches
            echo "Running qadst cluster..."
            qadst cluster \
                --input "$INPUT_FILE" \
                --filter \
                --min-samples "$min_samples" \
                --cluster-selection-epsilon "$epsilon" \
                $noise_option \
                --output-dir "$OUTPUT_DIR"

            # Check if clustering was successful
            if [ $? -ne 0 ]; then
                echo "Error: qadst cluster failed for run $RUN_NUMBER"
                echo "" >> "$REPORT_FILE"
                echo "~~~" >> "$REPORT_FILE"
                echo "# Run $RUN_NUMBER" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "## Options" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "- `--min-samples` $min_samples" >> "$REPORT_FILE"
                echo "- `--cluster-selection-epsilon` $epsilon" >> "$REPORT_FILE"
                echo "- `${noise_option}`" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "## Results" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "Clustering failed" >> "$REPORT_FILE"
                echo "~~~" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"

                # Increment run number and continue with the next combination
                RUN_NUMBER=$((RUN_NUMBER + 1))
                continue
            fi

            # Copy the output files to the run directory for preservation
            cp "$OUTPUT_DIR/qa_clusters.json" "$RUN_DIR/"
            cp "$OUTPUT_DIR/qa_cleaned.csv" "$RUN_DIR/"
            cp "$OUTPUT_DIR/engineering_questions.csv" "$RUN_DIR/" 2>/dev/null || true
            cp "$OUTPUT_DIR/filter_cache.json" "$RUN_DIR/" 2>/dev/null || true

            # Run qadst benchmark on the clustering results
            echo "Running qadst benchmark..."
            qadst benchmark \
                --clusters "$OUTPUT_DIR/qa_clusters.json" \
                --qa-pairs "$OUTPUT_DIR/qa_cleaned.csv" \
                --use-llm \
                --output-dir "$OUTPUT_DIR"

            # Check if benchmarking was successful
            if [ $? -ne 0 ]; then
                echo "Error: qadst benchmark failed for run $RUN_NUMBER"
                echo "" >> "$REPORT_FILE"
                echo "# Run $RUN_NUMBER" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "## Options" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "- \`--min-samples\` $min_samples" >> "$REPORT_FILE"
                echo "- \`--cluster-selection-epsilon\` $epsilon" >> "$REPORT_FILE"
                echo "- \`${noise_option}\`" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "## Results" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                echo "Benchmarking failed" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"

                # Increment run number and continue with the next combination
                RUN_NUMBER=$((RUN_NUMBER + 1))
                continue
            fi

            # Copy the benchmark results to the run directory
            cp "$OUTPUT_DIR/cluster_quality_report.csv" "$RUN_DIR/"

            # Copy the benchmark results to the report
            echo "" >> "$REPORT_FILE"
            echo "# Run $RUN_NUMBER" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "## Options" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "- \`--min-samples\` $min_samples" >> "$REPORT_FILE"
            echo "- \`--cluster-selection-epsilon\` $epsilon" >> "$REPORT_FILE"
            echo "- \`${noise_option}\`" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "## Results" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo '```csv' >> "$REPORT_FILE"
            cat "$OUTPUT_DIR/cluster_quality_report.csv" >> "$REPORT_FILE"
            echo '```' >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"

            # Increment run number
            RUN_NUMBER=$((RUN_NUMBER + 1))
        done
    done
done

echo "========================================================"
echo "Matrix test completed"
echo "Results saved to $RESULTS_DIR"
echo "Report saved to $REPORT_FILE"
echo "========================================================"

# Create a summary table of key metrics
echo "Creating summary table..."
echo "" >> "$REPORT_FILE"
echo "# Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Run | Min Samples | Epsilon | Noise Option | Clusters | Avg Coherence | Davies-Bouldin | Calinski-Harabasz | Silhouette |" >> "$REPORT_FILE"
echo "|-----|-------------|---------|--------------|----------|---------------|----------------|-------------------|------------|" >> "$REPORT_FILE"

# Reset run number for summary
RUN_NUMBER=1

for min_samples in "${MIN_SAMPLES_VALUES[@]}"; do
    for epsilon in "${EPSILON_VALUES[@]}"; do
        for noise_option in "${NOISE_OPTIONS[@]}"; do
            RUN_DIR="$RESULTS_DIR/run_$RUN_NUMBER"

            # Check if the run was successful
            if [ -f "$RUN_DIR/cluster_quality_report.csv" ]; then
                # Extract metrics from the last line of the CSV (summary row)
                SUMMARY=$(tail -n 1 "$RUN_DIR/cluster_quality_report.csv")

                # Parse the summary line to extract metrics
                # Format: Cluster_ID,Num_QA_Pairs,Coherence_Score,Topic_Label
                # The last line contains the summary with metrics in the Topic_Label field

                # Extract number of clusters (excluding noise)
                NUM_CLUSTERS=$(grep -v "SUMMARY" "$RUN_DIR/cluster_quality_report.csv" | wc -l)
                NUM_CLUSTERS=$((NUM_CLUSTERS - 1))  # Subtract header row

                # Extract metrics from the Topic_Label field
                TOPIC_LABEL=$(echo "$SUMMARY" | awk -F, '{print $4}')

                # Extract individual metrics using regex
                AVG_COHERENCE=$(echo "$TOPIC_LABEL" | grep -oP 'Avg Coherence: \K[0-9.]+' || echo "N/A")
                DAVIES_BOULDIN=$(echo "$TOPIC_LABEL" | grep -oP 'Davies-Bouldin: \K[0-9.]+' || echo "N/A")
                CALINSKI_HARABASZ=$(echo "$TOPIC_LABEL" | grep -oP 'Calinski-Harabasz: \K[0-9.]+' || echo "N/A")
                SILHOUETTE=$(echo "$TOPIC_LABEL" | grep -oP 'Silhouette: \K[0-9.-]+' || echo "N/A")

                # Add row to summary table
                echo "| $RUN_NUMBER | $min_samples | $epsilon | ${noise_option:2} | $NUM_CLUSTERS | $AVG_COHERENCE | $DAVIES_BOULDIN | $CALINSKI_HARABASZ | $SILHOUETTE |" >> "$REPORT_FILE"
            else
                # Add row with failure indication
                echo "| $RUN_NUMBER | $min_samples | $epsilon | ${noise_option:2} | Failed | - | - | - | - |" >> "$REPORT_FILE"
            fi

            # Increment run number
            RUN_NUMBER=$((RUN_NUMBER + 1))
        done
    done
done

echo "Summary table created"
echo "========================================================"
echo "Matrix test report complete"
echo "========================================================"

# Restore original output files
echo "Restoring original output files from backup"
cp "$BACKUP_DIR/qa_clusters.json" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BACKUP_DIR/qa_cleaned.csv" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BACKUP_DIR/engineering_questions.csv" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BACKUP_DIR/cluster_quality_report.csv" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BACKUP_DIR/filter_cache.json" "$OUTPUT_DIR/" 2>/dev/null || true

echo "========================================================"
echo "Original files restored"
echo "Matrix test complete"
echo "========================================================"
