"""Tests for the clusterer module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qadst.clusterer import HDBSCANQAClusterer


def test_cluster_method(mock_hdbscan_clusterer):
    """Test the cluster_method method."""
    assert mock_hdbscan_clusterer.cluster_method() == "hdbscan"


@patch("qadst.clusterer.HDBSCAN")
@patch("qadst.embeddings.get_embeddings_model")
@patch("qadst.base.ChatOpenAI")
def test_calculate_min_cluster_size(mock_chat, mock_embeddings, mock_hdbscan):
    """Test the _calculate_min_cluster_size method."""
    clusterer = HDBSCANQAClusterer(embedding_model_name="test")

    # Test with small dataset
    assert clusterer._calculate_min_cluster_size(10) == 5

    # Test with medium dataset
    assert clusterer._calculate_min_cluster_size(100) == 21

    # Test with large dataset
    assert clusterer._calculate_min_cluster_size(1000) == 47

    # Test with very large dataset
    assert clusterer._calculate_min_cluster_size(10000) == 84

    # Test with extremely large dataset (should be capped at 100)
    assert clusterer._calculate_min_cluster_size(100000) == 100


def test_identify_large_clusters(mock_hdbscan_clusterer):
    """Test the _identify_large_clusters method."""
    # Create test clusters
    clusters = {
        "1": {
            "questions": ["q1", "q2", "q3", "q4", "q5"],  # 5 questions (small)
            "qa_pairs": [{"question": "q1", "answer": "a1"} for _ in range(5)],
        },
        "2": {
            "questions": ["q" + str(i) for i in range(1, 21)],  # 20 questions (medium)
            "qa_pairs": [
                {"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 21)
            ],
        },
        "3": {
            "questions": ["q" + str(i) for i in range(1, 51)],  # 50 questions (large)
            "qa_pairs": [
                {"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 51)
            ],
        },
        "4": {
            "questions": [
                "q" + str(i) for i in range(1, 101)
            ],  # 100 questions (very large)
            "qa_pairs": [
                {"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 101)
            ],
        },
    }

    # Test with max_cluster_size = 30
    large_clusters = mock_hdbscan_clusterer._identify_large_clusters(clusters, 30)
    assert len(large_clusters) == 2
    assert "1" not in large_clusters
    assert "2" not in large_clusters
    assert "3" in large_clusters
    assert "4" in large_clusters

    # Test with max_cluster_size = 50
    large_clusters = mock_hdbscan_clusterer._identify_large_clusters(clusters, 50)
    assert len(large_clusters) == 1
    assert "1" not in large_clusters
    assert "2" not in large_clusters
    assert "3" not in large_clusters
    assert "4" in large_clusters

    # Test with max_cluster_size = 100
    large_clusters = mock_hdbscan_clusterer._identify_large_clusters(clusters, 100)
    assert len(large_clusters) == 0

    # Test with empty clusters
    large_clusters = mock_hdbscan_clusterer._identify_large_clusters({}, 30)
    assert len(large_clusters) == 0


def test_cluster_questions_empty_input(mock_hdbscan_clusterer):
    """Test the cluster_questions method with empty input."""
    result = mock_hdbscan_clusterer.cluster_questions([])
    assert result == {"clusters": []}


@patch("qadst.clusterer.HDBSCANQAClusterer._perform_hdbscan_clustering")
def test_cluster_questions_delegates_to_perform_hdbscan(
    mock_perform, mock_hdbscan_clusterer
):
    """Test that cluster_questions delegates to _perform_hdbscan_clustering."""
    # Setup mock return value
    expected_result = {"clusters": [{"id": 1, "representative": [], "source": []}]}
    mock_perform.return_value = expected_result

    # Call the method
    qa_pairs = [("test question", "test answer")]
    result = mock_hdbscan_clusterer.cluster_questions(qa_pairs)

    # Verify the mock was called with the right arguments
    mock_perform.assert_called_once_with(qa_pairs)

    # Verify the result
    assert result == expected_result


def test_get_recursive_hdbscan_params(mock_hdbscan_clusterer):
    """Test the _get_recursive_hdbscan_params method."""
    # Test with small cluster
    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(20)
    assert min_cluster_size == max(int(np.log(20) ** 1.5), 5)
    assert epsilon == 0.2

    # Test with medium cluster
    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(
        100
    )
    assert min_cluster_size == max(int(np.log(100) ** 1.5), 5)
    assert epsilon == 0.2

    # Test with large cluster
    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(
        500
    )
    assert min_cluster_size == max(int(np.log(500) ** 1.5), 5)
    assert epsilon == 0.2

    # Test with very small cluster (should use minimum of 5)
    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(5)
    assert min_cluster_size == 5
    assert epsilon == 0.2

    # Calculate expected values directly for verification
    expected_20 = max(int(np.log(20) ** 1.5), 5)
    expected_100 = max(int(np.log(100) ** 1.5), 5)
    expected_500 = max(int(np.log(500) ** 1.5), 5)
    expected_5 = 5

    # Verify the actual values for clarity
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(20) == (
        expected_20,
        0.2,
    )
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(100) == (
        expected_100,
        0.2,
    )
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(500) == (
        expected_500,
        0.2,
    )
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(5) == (expected_5, 0.2)


def test_calculate_kmeans_clusters(mock_hdbscan_clusterer):
    """Test the _calculate_kmeans_clusters method."""
    # Test with very small cluster (should return minimum of 2)
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(10) == 2
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(30) == 2
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(59) == 2

    # Test with medium clusters (should return num_questions / 30)
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(60) == 2
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(90) == 3
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(150) == 5
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(240) == 8

    # Test with large clusters (should be capped at 10)
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(300) == 10
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(500) == 10
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(1000) == 10

    # Verify the formula directly
    for num_questions in [10, 60, 90, 150, 300, 500]:
        expected = min(max(2, int(num_questions / 30)), 10)
        assert (
            mock_hdbscan_clusterer._calculate_kmeans_clusters(num_questions) == expected
        )


def test_create_subclusters(mock_hdbscan_clusterer):
    """Test the _create_subclusters method."""
    # Create test data
    cluster_id = "1"
    questions = ["q1", "q2", "q3", "q4", "q5"]
    qa_pairs = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"},
        {"question": "q4", "answer": "a4"},
        {"question": "q5", "answer": "a5"},
    ]

    # Test case 1: Normal clustering with 2 subclusters
    subcluster_labels = np.array([0, 0, 1, 1, 1])
    subclusters = mock_hdbscan_clusterer._create_subclusters(
        cluster_id, questions, qa_pairs, subcluster_labels
    )

    # Verify the subclusters
    assert len(subclusters) == 2
    assert "1.0" in subclusters
    assert "1.1" in subclusters

    # Verify the contents of subcluster 1.0
    assert len(subclusters["1.0"]["questions"]) == 2
    assert subclusters["1.0"]["questions"] == ["q1", "q2"]
    assert len(subclusters["1.0"]["qa_pairs"]) == 2
    assert subclusters["1.0"]["qa_pairs"] == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]

    # Verify the contents of subcluster 1.1
    assert len(subclusters["1.1"]["questions"]) == 3
    assert subclusters["1.1"]["questions"] == ["q3", "q4", "q5"]
    assert len(subclusters["1.1"]["qa_pairs"]) == 3
    assert subclusters["1.1"]["qa_pairs"] == [
        {"question": "q3", "answer": "a3"},
        {"question": "q4", "answer": "a4"},
        {"question": "q5", "answer": "a5"},
    ]

    # Test case 2: Clustering with noise points (label -1)
    subcluster_labels = np.array([0, -1, 1, -1, 1])
    subclusters = mock_hdbscan_clusterer._create_subclusters(
        cluster_id, questions, qa_pairs, subcluster_labels
    )

    # Verify the subclusters
    assert len(subclusters) == 2
    assert "1.0" in subclusters
    assert "1.1" in subclusters

    # Verify the contents of subcluster 1.0
    assert len(subclusters["1.0"]["questions"]) == 1
    assert subclusters["1.0"]["questions"] == ["q1"]
    assert len(subclusters["1.0"]["qa_pairs"]) == 1
    assert subclusters["1.0"]["qa_pairs"] == [{"question": "q1", "answer": "a1"}]

    # Verify the contents of subcluster 1.1
    assert len(subclusters["1.1"]["questions"]) == 2
    assert subclusters["1.1"]["questions"] == ["q3", "q5"]
    assert len(subclusters["1.1"]["qa_pairs"]) == 2
    assert subclusters["1.1"]["qa_pairs"] == [
        {"question": "q3", "answer": "a3"},
        {"question": "q5", "answer": "a5"},
    ]

    # Test case 3: All noise points
    subcluster_labels = np.array([-1, -1, -1, -1, -1])
    subclusters = mock_hdbscan_clusterer._create_subclusters(
        cluster_id, questions, qa_pairs, subcluster_labels
    )

    # Verify the subclusters (should be empty)
    assert len(subclusters) == 0


@patch("qadst.clusterer.HDBSCANQAClusterer._identify_large_clusters")
@patch("qadst.clusterer.HDBSCANQAClusterer._apply_recursive_clustering")
@patch("qadst.clusterer.HDBSCANQAClusterer._create_subclusters")
def test_handle_large_clusters(
    mock_create_subclusters,
    mock_apply_clustering,
    mock_identify_large,
    mock_hdbscan_clusterer,
):
    """Test the _handle_large_clusters method."""
    # Setup test data
    clusters = {
        "1": {
            "questions": ["q1", "q2", "q3"],
            "qa_pairs": [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
                {"question": "q3", "answer": "a3"},
            ],
        },
        "2": {
            "questions": ["q4", "q5", "q6", "q7", "q8"],
            "qa_pairs": [
                {"question": "q4", "answer": "a4"},
                {"question": "q5", "answer": "a5"},
                {"question": "q6", "answer": "a6"},
                {"question": "q7", "answer": "a7"},
                {"question": "q8", "answer": "a8"},
            ],
        },
    }
    total_questions = 8

    # Test case 1: No large clusters
    mock_identify_large.return_value = {}
    result = mock_hdbscan_clusterer._handle_large_clusters(
        clusters.copy(), total_questions
    )

    # Verify that identify_large_clusters was called with correct arguments
    mock_identify_large.assert_called_once()
    assert mock_identify_large.call_args[0][0] == clusters
    assert mock_identify_large.call_args[0][1] == max(int(total_questions * 0.2), 50)

    # Verify that no other methods were called
    mock_apply_clustering.assert_not_called()
    mock_create_subclusters.assert_not_called()

    # Verify that the clusters were returned unchanged
    assert result == clusters

    # Reset mocks for the next test
    mock_identify_large.reset_mock()
    mock_apply_clustering.reset_mock()
    mock_create_subclusters.reset_mock()

    # Test case 2: With large clusters
    large_clusters = {"2": clusters["2"]}
    mock_identify_large.return_value = large_clusters

    # Mock the get_embeddings method
    mock_hdbscan_clusterer.get_embeddings = MagicMock()
    mock_embeddings = [np.array([0.1, 0.2]) for _ in range(5)]
    mock_hdbscan_clusterer.get_embeddings.return_value = mock_embeddings

    # Mock the _apply_recursive_clustering method
    subcluster_labels = np.array([0, 0, 1, 1, 1])
    mock_apply_clustering.return_value = (subcluster_labels, 2)

    # Mock the _create_subclusters method
    subclusters = {
        "2.0": {
            "questions": ["q4", "q5"],
            "qa_pairs": [
                {"question": "q4", "answer": "a4"},
                {"question": "q5", "answer": "a5"},
            ],
        },
        "2.1": {
            "questions": ["q6", "q7", "q8"],
            "qa_pairs": [
                {"question": "q6", "answer": "a6"},
                {"question": "q7", "answer": "a7"},
                {"question": "q8", "answer": "a8"},
            ],
        },
    }
    mock_create_subclusters.return_value = subclusters

    # Call the method
    result = mock_hdbscan_clusterer._handle_large_clusters(
        clusters.copy(), total_questions
    )

    # Verify that identify_large_clusters was called with correct arguments
    mock_identify_large.assert_called_once()

    # Verify that get_embeddings was called with the questions from the large cluster
    mock_hdbscan_clusterer.get_embeddings.assert_called_once()
    args, kwargs = mock_hdbscan_clusterer.get_embeddings.call_args
    assert args[0] == clusters["2"]["questions"]

    # Verify that _apply_recursive_clustering was called with correct arguments
    mock_apply_clustering.assert_called_once()
    assert mock_apply_clustering.call_args[0][0] == clusters["2"]["questions"]
    assert np.array_equal(
        mock_apply_clustering.call_args[0][1], np.array(mock_embeddings)
    )
    assert mock_apply_clustering.call_args[0][2] == "2"

    # Verify that _create_subclusters was called with correct arguments
    mock_create_subclusters.assert_called_once()
    assert mock_create_subclusters.call_args[0][0] == "2"
    assert mock_create_subclusters.call_args[0][1] == clusters["2"]["questions"]
    assert mock_create_subclusters.call_args[0][2] == clusters["2"]["qa_pairs"]
    assert np.array_equal(mock_create_subclusters.call_args[0][3], subcluster_labels)

    # Verify the result
    expected_result = {
        "1": clusters["1"],
        "2.0": subclusters["2.0"],
        "2.1": subclusters["2.1"],
    }
    assert result == expected_result


@patch("qadst.clusterer.HDBSCAN")
@patch("qadst.embeddings.get_embeddings_model")
@patch("qadst.base.ChatOpenAI")
def test_custom_hdbscan_parameters(mock_chat, mock_embeddings, mock_hdbscan):
    """Test that custom HDBSCAN parameters are used."""
    # Create a clusterer with custom parameters
    clusterer = HDBSCANQAClusterer(
        embedding_model_name="test",
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_epsilon=0.2,
    )

    # Check that the parameters were set correctly
    assert clusterer.min_cluster_size == 10
    assert clusterer.min_samples == 5
    assert clusterer.cluster_selection_epsilon == 0.2


@patch("qadst.clusterer.HDBSCAN")
@patch("qadst.embeddings.get_embeddings_model")
@patch("qadst.base.ChatOpenAI")
def test_cluster_selection_method_parameter(mock_chat, mock_embeddings, mock_hdbscan):
    """Test that the cluster_selection_method parameter is used."""
    # Create a clusterer with custom cluster_selection_method
    clusterer = HDBSCANQAClusterer(
        embedding_model_name="test",
        cluster_selection_method="leaf",
    )

    # Check that the parameter was set correctly
    assert clusterer.cluster_selection_method == "leaf"


@patch("qadst.clusterer.HDBSCAN")
@patch("qadst.embeddings.get_embeddings_model")
@patch("qadst.base.ChatOpenAI")
def test_keep_noise_parameter(mock_chat, mock_embeddings, mock_hdbscan):
    """Test that the keep_noise parameter is used."""
    # Create a clusterer with keep_noise=True
    clusterer = HDBSCANQAClusterer(
        embedding_model_name="test",
        keep_noise=True,
    )

    # Check that the parameter was set correctly
    assert clusterer.keep_noise is True


@patch("qadst.clusterer.HDBSCAN")
@patch("qadst.embeddings.get_embeddings_model")
@patch("qadst.base.ChatOpenAI")
def test_cluster_noise_points(mock_chat, mock_embeddings, mock_hdbscan):
    """Test the _cluster_noise_points method."""
    # Create the clusterer with a mock model name
    clusterer = HDBSCANQAClusterer(embedding_model_name="test")

    # Set up a mock embeddings model directly
    mock_embeddings_model = MagicMock()
    mock_embeddings_model.embed_documents.return_value = np.array(
        [[0.1, 0.2], [0.3, 0.4]]
    )
    clusterer.embeddings_model = mock_embeddings_model

    # Mock KMeans
    with patch("qadst.clusterer.KMeans") as mock_kmeans:
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.labels_ = np.array([0, 1])
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1])
        mock_kmeans.return_value = mock_kmeans_instance

        # Call the method
        result = clusterer._cluster_noise_points(
            [("question1", "answer1"), ("question2", "answer2")], 2
        )

        # Check that KMeans was called with the right parameters
        mock_kmeans.assert_called_once()
        assert mock_kmeans.call_args[1]["n_clusters"] == 2

        # Create a mock for HDBSCAN
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array([0])
        mock_hdbscan_class.return_value = mock_hdbscan_instance

        # Call the method that uses HDBSCAN
        clusterer._perform_hdbscan_clustering([("test question", "test answer")])

        # Check that HDBSCAN was called with the correct parameters
        mock_hdbscan_class.assert_called_once_with(
            min_cluster_size=50,
            min_samples=3,
            cluster_selection_epsilon=0.2,
            cluster_selection_method="eom",  # Default value
        )


def test_cluster_selection_method_parameter():
    """Test that the cluster_selection_method parameter is used correctly."""
    with (
        patch("qadst.clusterer.HDBSCAN") as mock_hdbscan_class,
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create clusterer with custom cluster_selection_method
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
            cluster_selection_method="leaf",
        )

        # Mock the get_embeddings method
        clusterer.get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Create a mock for HDBSCAN
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array([0])
        mock_hdbscan_class.return_value = mock_hdbscan_instance

        # Call the method that uses HDBSCAN
        clusterer._perform_hdbscan_clustering([("test question", "test answer")])

        # Check that HDBSCAN was called with the correct parameters
        # min_cluster_size will be calculated automatically
        # min_samples and cluster_selection_epsilon will use defaults
        mock_hdbscan_class.assert_called_once_with(
            min_cluster_size=ANY,  # Calculated automatically
            min_samples=5,  # Default value
            cluster_selection_epsilon=0.3,  # Default value
            cluster_selection_method="leaf",  # Custom value
        )


def test_keep_noise_parameter():
    """Test that the keep_noise parameter preserves noise points."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create a clusterer with keep_noise=True
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
            keep_noise=True,
        )

        # Mock the get_embeddings method
        clusterer.get_embeddings = MagicMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )

        # Create a mock for HDBSCAN
        mock_hdbscan_instance = MagicMock()
        # Set up the mock to return one regular cluster and one noise point
        mock_hdbscan_instance.fit_predict.return_value = np.array([0, 0, -1])

        # Replace the HDBSCAN class with a mock
        with patch("qadst.clusterer.HDBSCAN", return_value=mock_hdbscan_instance):
            # Call the clustering method with test data
            qa_pairs = [
                ("question1", "answer1"),
                ("question2", "answer2"),
                ("question3", "answer3"),
            ]
            result = clusterer._perform_hdbscan_clustering(qa_pairs)

            # Verify that the result contains both a regular cluster and a noise cluster
            assert len(result["clusters"]) == 2

            # Find the noise cluster (id should be 0)
            noise_cluster = None
            regular_cluster = None
            for cluster in result["clusters"]:
                if cluster.get("is_noise", False):
                    noise_cluster = cluster
                else:
                    regular_cluster = cluster

            # Verify the noise cluster exists and has the correct structure
            assert noise_cluster is not None
            assert noise_cluster["id"] == 0
            assert len(noise_cluster["representative"]) == 0
            assert len(noise_cluster["source"]) == 1
            assert noise_cluster["source"][0]["question"] == "question3"

            # Verify the regular cluster
            assert regular_cluster is not None
            assert regular_cluster["id"] == 1
            assert len(regular_cluster["source"]) == 2
            assert regular_cluster["source"][0]["question"] == "question1"
            assert regular_cluster["source"][1]["question"] == "question2"


def test_cluster_noise_points():
    """Test that the _cluster_noise_points method is called when keep_noise is False."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create a clusterer with keep_noise=False
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
            keep_noise=False,
        )

        # Test the _cluster_noise_points method directly
        noise_qa_pairs = [
            ("question1", "answer1"),
            ("question2", "answer2"),
        ]

        # Mock the embeddings_model.embed_documents method
        clusterer.embeddings_model = MagicMock()
        clusterer.embeddings_model.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        # Mock KMeans
        with patch("qadst.clusterer.KMeans") as mock_kmeans_class:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 1])
            mock_kmeans_class.return_value = mock_kmeans_instance

            # Call the method
            result = clusterer._cluster_noise_points(noise_qa_pairs, 1)

            # Verify KMeans was called
            mock_kmeans_class.assert_called_once()

            # Verify the result structure
            assert len(result) == 2
            assert "0" in result
            assert "1" in result
            assert result["0"]["questions"] == ["question1"]
            assert result["1"]["questions"] == ["question2"]


@pytest.mark.parametrize(
    "fn, input_str, expected_result",
    [
        # Valid conversions
        (int, "123", 123),
        (int, "-123", -123),
        (int, "0", 0),
        (float, "123.45", 123.45),
        (float, "-123.45", -123.45),
        (float, "0.0", 0.0),
        (float, ".5", 0.5),
        (float, "1e6", 1000000.0),
        (float, "-1e6", -1000000.0),
        (complex, "1+2j", 1 + 2j),
        (complex, "-1-2j", -1 - 2j),
        # Invalid conversions
        (int, "123.45", None),
        (int, "abc", None),
        (int, "", None),
        (int, "1e6", None),
        (float, "abc", None),
        (float, "", None),
        (float, "1+2j", None),
        (complex, "abc", None),
        (complex, "", None),
    ],
)
def test_if_ok(fn, input_str, expected_result):
    """Test the _if_ok method with various inputs."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        result = clusterer._if_ok(fn, input_str)

        if expected_result is None:
            assert result is None
        else:
            assert result == expected_result


@pytest.mark.parametrize(
    "input_str, expected_result",
    [
        # Valid numeric strings
        ("123", True),
        ("-123", True),
        ("0", True),
        ("123.45", True),
        ("-123.45", True),
        ("0.0", True),
        (".5", True),
        ("1e6", True),
        ("-1e6", True),
        ("1+2j", True),
        ("-1-2j", True),
        ("inf", True),
        ("-inf", True),
        ("nan", True),
        # Invalid numeric strings
        ("abc", False),
        ("", False),
        ("123abc", False),
        ("abc123", False),
        ("12.34.56", False),
        ("12,345", False),
        ("$123", False),
        ("None", False),
        ("True", False),
        ("False", False),
    ],
)
def test_is_numeric(input_str, expected_result):
    """Test the _is_numeric method with various inputs."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        result = clusterer._is_numeric(input_str)
        assert result is expected_result


@pytest.mark.parametrize(
    "cluster_id, expected_result, expected_range",
    [
        # Regular cluster IDs
        ("0", 1, None),
        ("1", 2, None),
        ("2", 3, None),
        ("10", 11, None),
        ("100", 101, None),
        ("-1", 0, None),
        ("-2", -1, None),
        # Subcluster IDs with numeric components
        ("0.0", 1, None),
        ("0.1", 2, None),
        ("1.0", 1001, None),
        ("1.1", 1002, None),
        ("1.2", 1003, None),
        ("2.0", 2001, None),
        ("10.5", 10006, None),
        ("-1.0", -999, None),
        ("-1.1", -998, None),
        # Subcluster IDs with non-numeric components
        ("a.0", None, (1000, 11000)),
        ("0.a", None, (1000, 11000)),
        ("a.b", None, (1000, 11000)),
        # Non-numeric IDs
        ("a", None, (1, 1001)),
        ("abc", None, (1, 1001)),
        ("", None, (1, 1001)),
        ("None", None, (1, 1001)),
        ("True", None, (1, 1001)),
        ("False", None, (1, 1001)),
        # Edge cases
        ("1.0.0", None, (1000, 11000)),
        ("1,000", None, (1, 1001)),
        ("1e6", 1000001, None),
        ("1.0e6", None, (1000, 11000)),
    ],
)
def test_convert_cluster_id_to_numeric(cluster_id, expected_result, expected_range):
    """Test the _convert_cluster_id_to_numeric method with various inputs."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        result = clusterer._convert_cluster_id_to_numeric(cluster_id)

        # Always ensure we get an integer
        assert isinstance(result, int)

        if expected_result is not None:
            # For cases where we expect a specific value
            assert result == expected_result
        elif expected_range is not None:
            # For cases where we expect a value in a range
            min_val, max_val = expected_range
            assert min_val <= result < max_val
        else:
            # Should never happen with our test cases
            assert (
                False
            ), "Test case must specify either expected_result or expected_range"
        # Create a mock for HDBSCAN
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array([0])
        mock_hdbscan_class.return_value = mock_hdbscan_instance

        # Call the method that uses HDBSCAN
        clusterer._perform_hdbscan_clustering([("test question", "test answer")])

        # Check that HDBSCAN was called with the correct parameters
        mock_hdbscan_class.assert_called_once_with(
            min_cluster_size=50,
            min_samples=3,
            cluster_selection_epsilon=0.2,
            cluster_selection_method="eom",  # Default value
        )


def test_cluster_selection_method_parameter():
    """Test that the cluster_selection_method parameter is used correctly."""
    with (
        patch("qadst.clusterer.HDBSCAN") as mock_hdbscan_class,
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create clusterer with custom cluster_selection_method
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
            cluster_selection_method="leaf",
        )

        # Mock the get_embeddings method
        clusterer.get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Create a mock for HDBSCAN
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array([0])
        mock_hdbscan_class.return_value = mock_hdbscan_instance

        # Call the method that uses HDBSCAN
        clusterer._perform_hdbscan_clustering([("test question", "test answer")])

        # Check that HDBSCAN was called with the correct parameters
        # min_cluster_size will be calculated automatically
        # min_samples and cluster_selection_epsilon will use defaults
        mock_hdbscan_class.assert_called_once_with(
            min_cluster_size=ANY,  # Calculated automatically
            min_samples=5,  # Default value
            cluster_selection_epsilon=0.3,  # Default value
            cluster_selection_method="leaf",  # Custom value
        )


def test_keep_noise_parameter():
    """Test that the keep_noise parameter preserves noise points."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create a clusterer with keep_noise=True
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
            keep_noise=True,
        )

        # Mock the get_embeddings method
        clusterer.get_embeddings = MagicMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )

        # Create a mock for HDBSCAN
        mock_hdbscan_instance = MagicMock()
        # Set up the mock to return one regular cluster and one noise point
        mock_hdbscan_instance.fit_predict.return_value = np.array([0, 0, -1])

        # Replace the HDBSCAN class with a mock
        with patch("qadst.clusterer.HDBSCAN", return_value=mock_hdbscan_instance):
            # Call the clustering method with test data
            qa_pairs = [
                ("question1", "answer1"),
                ("question2", "answer2"),
                ("question3", "answer3"),
            ]
            result = clusterer._perform_hdbscan_clustering(qa_pairs)

            # Verify that the result contains both a regular cluster and a noise cluster
            assert len(result["clusters"]) == 2

            # Find the noise cluster (id should be 0)
            noise_cluster = None
            regular_cluster = None
            for cluster in result["clusters"]:
                if cluster.get("is_noise", False):
                    noise_cluster = cluster
                else:
                    regular_cluster = cluster

            # Verify the noise cluster exists and has the correct structure
            assert noise_cluster is not None
            assert noise_cluster["id"] == 0
            assert len(noise_cluster["representative"]) == 0
            assert len(noise_cluster["source"]) == 1
            assert noise_cluster["source"][0]["question"] == "question3"

            # Verify the regular cluster
            assert regular_cluster is not None
            assert regular_cluster["id"] == 1
            assert len(regular_cluster["source"]) == 2
            assert regular_cluster["source"][0]["question"] == "question1"
            assert regular_cluster["source"][1]["question"] == "question2"


def test_cluster_noise_points():
    """Test that the _cluster_noise_points method is called when keep_noise is False."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create a clusterer with keep_noise=False
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
            keep_noise=False,
        )

        # Test the _cluster_noise_points method directly
        noise_qa_pairs = [
            ("question1", "answer1"),
            ("question2", "answer2"),
        ]

        # Mock the embeddings_model.embed_documents method
        clusterer.embeddings_model = MagicMock()
        clusterer.embeddings_model.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        # Mock KMeans
        with patch("qadst.clusterer.KMeans") as mock_kmeans_class:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 1])
            mock_kmeans_class.return_value = mock_kmeans_instance

            # Call the method
            result = clusterer._cluster_noise_points(noise_qa_pairs, 1)

            # Verify KMeans was called
            mock_kmeans_class.assert_called_once()

            # Verify the result structure
            assert len(result) == 2
            assert "0" in result
            assert "1" in result
            assert result["0"]["questions"] == ["question1"]
            assert result["1"]["questions"] == ["question2"]
