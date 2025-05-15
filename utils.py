import configparser
import pandas as pd
import os
import logging
import json # Pour parser les dictionnaires/listes dans le config
from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict

# Configuration du logging (peut être configuré au niveau de l'application principale)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Optionnel: Définir des structures pour les retours de config pour plus de clarté
class GptConfig(TypedDict):
    file_path: str
    output_file: str
    columns_csv: List[str]

class KmeansConfig(TypedDict):
    file_path: str
    output_file: str
    columns_csv: List[str]
    properties_dict: Dict[str, int]
    prop_columns: List[str]
    num_clusters: int

def load_config(script_dir: str, config_file_name: str) -> Optional[Union[GptConfig, KmeansConfig]]:
    """
    Loads configuration from a specified file.

    Args:
        script_dir: The directory where the script is located.
        config_file_name: The name of the configuration file (e.g., 'config_gpt.txt').

    Returns:
        A dictionary representing the configuration (GptConfig or KmeansConfig)
        or None if loading fails.
    """
    config_file = os.path.join(script_dir, 'configs', config_file_name)
    config = configparser.ConfigParser()

    if not os.path.exists(config_file):
        logging.error(f"Configuration file not found: {config_file}")
        return None

    try:
        config.read(config_file)

        # --- Configuration Commune ---
        try:
            file_path = config.get('base', 'file_path')
            output_file = config.get('base', 'output_file')
            # Utiliser json.loads pour lire les listes de colonnes
            columns_csv_str = config.get('columns_csv', 'cols', fallback='[]') # Fallback au cas où
            columns_csv = json.loads(columns_csv_str)
            if not isinstance(columns_csv, list):
                 raise ValueError("columns_csv:cols should be a JSON list.")

        except (configparser.NoSectionError, configparser.NoOptionError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error reading common configuration sections in {config_file_name}: {e}")
            return None

        # --- Configuration Spécifique ---
        if config_file_name == 'config_gpt.txt':
            # Pas d'autres clés spécifiques pour GPT dans l'exemple original
            return GptConfig(
                file_path=file_path,
                output_file=output_file,
                columns_csv=columns_csv
            )
        else: # Supposons que c'est pour K-Means ou autre
            try:
                num_clusters = config.getint('base', 'num_clusters')

                # Lire le dictionnaire de propriétés en utilisant JSON
                properties_str = config.get('properties_dict', 'mapping', fallback='{}') # Clé 'mapping' attendue
                properties_dict_raw = json.loads(properties_str)
                # Assurer que les valeurs sont bien des entiers
                properties_dict = {str(k): int(v) for k, v in properties_dict_raw.items()}

                # Lire les colonnes de propriétés en utilisant JSON
                prop_columns_str = config.get('prop_columns', 'prop_columns', fallback='[]')
                prop_columns = json.loads(prop_columns_str)
                if not isinstance(prop_columns, list):
                    raise ValueError("prop_columns:prop_columns should be a JSON list.")

                return KmeansConfig(
                    file_path=file_path,
                    output_file=output_file,
                    columns_csv=columns_csv,
                    properties_dict=properties_dict,
                    prop_columns=prop_columns,
                    num_clusters=num_clusters
                )
            except (configparser.NoSectionError, configparser.NoOptionError, json.JSONDecodeError, ValueError, TypeError) as e:
                logging.error(f"Error reading K-Means specific configuration in {config_file_name}: {e}")
                return None

    except configparser.Error as e:
        logging.error(f"Error parsing configuration file {config_file_name}: {e}")
        return None

def load_data(file_path: str, usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file.

    Args:
        file_path: Path to the CSV file.
        usecols: Optional list of columns to read.

    Returns:
        A pandas DataFrame with the loaded data, or None if an error occurs.
    """
    try:
        data = pd.read_csv(file_path, sep=";", usecols=usecols)
        logging.info(f"Successfully loaded data from {file_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Data file is empty: {file_path}")
        return None
    except pd.errors.ParserError as e:
         logging.error(f"Error parsing CSV file {file_path}: {e}")
         return None
    except ValueError as e:
        # Peut arriver si usecols contient une colonne inexistante
        logging.error(f"Error reading CSV {file_path} (check columns?): {e}")
        return None
    except Exception as e: # Capture générique pour d'autres erreurs imprévues
        logging.error(f"An unexpected error occurred while loading data from {file_path}: {e}", exc_info=True)
        return None

