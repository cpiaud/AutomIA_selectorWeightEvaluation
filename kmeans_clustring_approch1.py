# import packages
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter, OrderedDict
import configparser
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

# Assuming utils.py exists and provides these functions
try:
    from utils import load_config, load_data
    # Import the config types for type checking
    from utils import KmeansConfig, GptConfig # Assuming GptConfig might be used elsewhere or for future proofing
except ImportError as e:
    print("Error: utils.py not found or contains errors.")
    logging.error(f"Failed to import from utils: {e}")
    # Define dummy functions if utils is missing, for basic script analysis

    def load_data(file_path: str, usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        print("Warning: Using dummy load_data")
        # Return a dummy DataFrame for structure
        if file_path == "dummy_path.csv":
             return pd.DataFrame({'col1': [1, 2, 3], 'prop1': ["A", "B", "A"]})
        return None
    # Define dummy config loader if utils is missing, returning None to indicate failure
    def load_config(script_dir: str, config_filename: str) -> None:
        print("Warning: Using dummy load_config returning None")
        return None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 1 # Specific initialization strategy might require n_init=1

# current folder
script_dir = os.path.dirname(os.path.abspath(__file__))

def preprocess_properties(data: pd.DataFrame, properties_map: Dict[Any, int], prop_columns: List[str]) -> pd.DataFrame:
    """
    Transforms property values in a DataFrame using the provided mapping dictionary.

    Args:
        data: DataFrame containing the data to transform.
        properties_map: Dictionary mapping original property values to desired integer values.
        prop_columns: List of column names in 'data' containing properties to transform.

    Returns:
        The DataFrame with transformed properties. Returns a copy to avoid side effects.

    Raises:
        KeyError: If a column in prop_columns is not found in the DataFrame.
        ValueError: If mapping leads to unexpected types or issues.
    """
    data_copy = data.copy()
    try:
        for prop in prop_columns:
            if prop not in data_copy.columns:
                 raise KeyError(f"Property column '{prop}' not found in DataFrame.")
            # Use .map for transformation. Unmapped values will become NaN.
            data_copy[prop] = data_copy[prop].map(properties_map)
            # Optional: Handle potential NaNs if mapping is not exhaustive
            if data_copy[prop].isnull().any():
                logging.warning(f"Column '{prop}' contains NaN values after mapping. "
                                f"Ensure properties_map covers all values or handle NaNs explicitly.")
                # Example: Fill NaN with a specific value if needed
                # data_copy[prop] = data_copy[prop].fillna(some_default_value)
    except Exception as e:
        logging.error(f"Error during property preprocessing: {e}")
        raise ValueError(f"Failed to preprocess properties: {e}") from e
    return data_copy

def apply_kmeans(data: pd.DataFrame, prop_columns: List[str], num_clusters: int) -> Tuple[List[Dict[int, float]], Dict[str, pd.Series]]:
    """
    Applies K-means clustering column-wise on specified DataFrame columns and calculates cluster weights.

    Args:
        data: DataFrame containing the data to analyze. Specified columns must be numeric.
        prop_columns: List of column names to apply K-means on.
        num_clusters: Number of clusters for K-means.

    Returns:
        A tuple containing:
        - list_dict_weights: List of dictionaries, each holding cluster weights (percentage) for a specific column.
        - cluster_labels: Dictionary mapping column name to the Series of cluster labels.

    Raises:
        KeyError: If a column in prop_columns is not found in the DataFrame.
        ValueError: If data in a property column is not suitable for K-means (e.g., non-numeric, NaN).
    """
    # Initialize cluster centers based on the assumption that properties are mapped 0 to num_clusters-1
    # This forces initial centroids to these specific values.
    initial_centers = [[i] for i in range(num_clusters)]
    logging.info(f"Using initial centers for KMeans: {initial_centers}")

    list_dict_weights: List[Dict[int, float]] = []
    cluster_labels: Dict[str, pd.Series] = {}

    for prop in prop_columns:
        if prop not in data.columns:
            raise KeyError(f"Column '{prop}' not found in DataFrame for K-means.")

        column_data = data[[prop]] # Select column as DataFrame
        # Check for non-numeric types or NaNs before applying KMeans
        if not pd.api.types.is_numeric_dtype(column_data[prop]):
             raise ValueError(f"Column '{prop}' must be numeric for K-means.")
        if column_data.isnull().any().any():
             raise ValueError(f"Column '{prop}' contains NaN values, cannot apply K-means. Handle NaNs first.")

        try:
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=KMEANS_RANDOM_STATE,
                n_init=KMEANS_N_INIT, # Use constant
                init=initial_centers # Specific initialization
            )
            labels = kmeans.fit_predict(column_data)
            cluster_labels[f'cluster_{prop}'] = pd.Series(labels, index=data.index)

            cluster_counts = Counter(labels)
            total = sum(cluster_counts.values())
            if total == 0:
                 weights = {k: 0.0 for k in range(num_clusters)} # Handle empty column case
            else:
                # Calculate weights as percentages
                weights = {k: round((v / total) * 100.0) for k, v in cluster_counts.items()}

            # Ensure all clusters (0 to num_clusters-1) are present in the weights dict, even if empty
            full_weights = {i: weights.get(i, 0.0) for i in range(num_clusters)}
            list_dict_weights.append(full_weights)
            logging.info(f"Calculated weights for '{prop}': {full_weights}")

        except Exception as e:
            logging.error(f"Error applying K-means to column '{prop}': {e}")
            raise ValueError(f"K-means failed for column '{prop}': {e}") from e

    return list_dict_weights, cluster_labels


def combine_weights(list_weights: List[Dict[int, float]]) -> OrderedDict[int, float]:
    """
    Combines cluster weights from multiple dictionaries into a single sorted dictionary.

    Args:
        list_weights: List of dictionaries, where each dictionary contains cluster weights for a specific column.
                      Keys are cluster indices (int), values are weights (float).

    Returns:
        A dictionary sorted by cluster index (key) with combined weights.
    """
    # Use Counter to sum weights for the same cluster index across dictionaries
    combined_weights: Counter[int] = sum((Counter(d) for d in list_weights), Counter())

    # Sort by cluster index (key) and return as OrderedDict (or dict in Python 3.7+)
    # Using OrderedDict explicitly shows the intent for ordering.
    sorted_weights = OrderedDict(sorted(combined_weights.items()))
    logging.info(f"Combined and sorted weights: {sorted_weights}")
    return sorted_weights


def write_to_properties_file(
    final_weights: Dict[int, float],
    output_map: Dict[str, int],
    file_path: str,
    key_to_exclude: Optional[str] = None
) -> None:
    """
    Writes the final weights to a properties file using a provided mapping.

    Args:
        final_weights: Dictionary containing the final combined weights (cluster_index -> weight).
        output_map: Dictionary mapping the desired output keys (e.g., original property values)
                    to the cluster indices used as keys in final_weights.
        file_path: Path to the output properties file.
        key_to_exclude: Optional key from output_map to exclude from the output file.

    Raises:
        IOError: If the file cannot be written.
        KeyError: If a value from output_map is not found as a key in final_weights (unless excluded).
    """
    # Prepare the dictionary for the properties file
    # Keys are from output_map (e.g., 'A', 'B'), values are looked up in final_weights using output_map's values (0, 1, ...)
    final_dict_poids: Dict[str, float] = {}
    for key, cluster_index in output_map.items():
        if key == key_to_exclude:
            logging.info(f"Excluding key '{key}' from output properties file.")
            continue
        try:
            final_dict_poids[key] = final_weights[cluster_index]
        except KeyError:
            # Handle cases where a cluster index expected by the map doesn't have a weight
            # This might happen if a cluster was empty across all properties.
            logging.warning(f"Cluster index '{cluster_index}' (mapped by key '{key}') not found in final_weights. Setting weight to 0.")
            final_dict_poids[key] = 0.0 # Assign a default value, e.g., 0

    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Failed to create output directory '{output_dir}': {e}")
            raise IOError(f"Cannot create output directory '{output_dir}'") from e

    # Write to the file
    try:
        with open(file_path, 'w') as file:
            for key, value in final_dict_poids.items():
                # Convertir la valeur en entier avant de l'écrire
                file.write(f"{key}={int(value)}\n")
        logging.info(f"Successfully wrote weights to {file_path}")
    except IOError as e:
        logging.error(f"Failed to write to properties file '{file_path}': {e}")
        raise IOError(f"Cannot write to file '{file_path}'") from e


def main():
    """
    Main function to execute the K-means clustering workflow.
    """
    config_filename = 'config_kmeans.txt'
    weights_subdir = 'weights'
    key_to_exclude_from_output = 'n' # Make exclusion explicit

    try:
        # Load configuration
        config_data = load_config(script_dir, config_filename)

        # Check if configuration loading was successful
        if config_data is None:
            logging.error("Failed to load configuration. Exiting.")
            return
        # Check if the loaded config is the expected type for this script
        if not isinstance(config_data, dict) or 'num_clusters' not in config_data: # Basic check for KmeansConfig structure
            logging.error(f"Loaded configuration is not the expected K-Means config format. Exiting.")
            return

        # Extract values from the config dictionary
        file_path: str = config_data['file_path']
        columns_csv: List[str] = config_data['columns_csv']
        properties_map: Dict[str, int] = config_data['properties_dict']
        prop_columns: List[str] = config_data['prop_columns']
        num_clusters: int = config_data['num_clusters']
        output_filename: str = config_data['output_file']
        logging.info("Configuration loaded successfully.")
        logging.info(f"Input file: {file_path}")
        logging.info(f"Property columns for K-means: {prop_columns}")
        logging.info(f"Number of clusters: {num_clusters}")
        logging.info(f"Output file: {output_filename}")

        # Load data
        data = load_data(file_path, usecols=columns_csv)
        if data is None:
            logging.error(f"Failed to load data from {file_path}. Exiting.")
            return
        logging.info(f"Data loaded successfully. Shape: {data.shape}")

        # Preprocess properties (map to integers)
        data_processed = preprocess_properties(data, properties_map, prop_columns)
        logging.info("Property preprocessing completed.")

        # Apply K-means and calculate weights
        list_weights, cluster_labels_dict = apply_kmeans(data_processed, prop_columns, num_clusters)

        # Combine weights from different properties
        final_weights = combine_weights(list_weights)

        # Define the output path correctly
        output_file_path = os.path.join(script_dir, weights_subdir, output_filename)

        # Write weights to the properties file
        # We use properties_map here assuming it maps output keys (like 'A', 'B') to cluster indices (0, 1)
        write_to_properties_file(final_weights, properties_map, output_file_path, key_to_exclude_from_output)

        # Optional: Add cluster labels back to the original DataFrame for inspection
        # data_with_clusters = data.join(pd.DataFrame(cluster_labels_dict))
        # logging.info("Final DataFrame with cluster labels:")
        # print(data_with_clusters.head()) # Print head instead of full df

        logging.info("--- Results ---")
        logging.info(f"Individual Weights per Property: {list_weights}")
        logging.info(f"Combined Final Weights (Cluster Index: Weight): {final_weights}")
        logging.info(f"Output written to: {output_file_path}")
        logging.info("Script finished successfully.")

    except FileNotFoundError as e:
        logging.error(f"Configuration or data file not found: {e}")
    except TypeError as e:
        logging.error(f"Type error during configuration processing or execution: {e}. Check config structure and usage.")
    except KeyError as e:
        logging.error(f"Data processing error: Missing key {e}")
    except ValueError as e:
        logging.error(f"Data processing error: Invalid value or data type - {e}")
    except IOError as e:
        logging.error(f"File input/output error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback


if __name__ == "__main__":
    main()
