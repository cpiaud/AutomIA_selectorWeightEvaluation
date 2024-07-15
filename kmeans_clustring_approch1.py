import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter, OrderedDict
import configparser
import os
def load_config(file_path):
   
    config = configparser.ConfigParser()
    config.read(file_path)
    
    file_path = config.get('base', 'file_path')
    output_file = config.get('base', 'output_file')
    num_clusters = config.getint('base', 'num_clusters')
    properties_str = config['properties_dict']
    print(properties_str.items())
    properties_dict = {}
    for key, value in properties_str.items():
        cleaned_key = key.strip().strip("'")
        cleaned_value = int(value.strip().rstrip(','))
        properties_dict[cleaned_key] = cleaned_value

    prop_columns = config.get('prop_columns', 'prop_columns').split(', ')
    columns_csv=config.get('columns_csv', 'cols').split(', ')
    return file_path,columns_csv, properties_dict, prop_columns, num_clusters, output_file


def load_data(file_path,usecols):
    try:
        data = pd.read_csv(file_path, sep=";", usecols=usecols)
        return data
    except FileNotFoundError:
        print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        return None

def preprocess_properties(data, properties_dict,prop_columns):
    """ Transforme les propriétés en utilisant le dictionnaire de propriétés fourni. """
    for prop in prop_columns:
        data[prop] = data[prop].map(properties_dict)
    return data

def apply_kmeans(data, prop_columns, num_clusters):
    #Applique l'algorithme K-means 
    initial_centers = [[i] for i in range(num_clusters)]
    list_dict_weights = []

    for prop in prop_columns:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1, init=initial_centers)
        data[f'cluster_{prop}'] = kmeans.fit_predict(data[[prop]])
        
        cluster_counts = Counter(data[f'cluster_{prop}'])
        total = sum(cluster_counts.values())
        weights = {k: round(v / total * 100) for k, v in cluster_counts.items()}
        list_dict_weights.append(weights)
    
    return list_dict_weights

def combine_weights(list_weights):
    #Combine et trie les poids de tous les clusters.
    combined_weights = sum((Counter(d) for d in list_weights), Counter())
    sorted_weights = OrderedDict(sorted(combined_weights.items()))
    return sorted_weights

def write_to_properties_file(final_dict,properties_dict , file_path):
    #Écrit le dictionnaire final dans un fichier de propriétés selon le format spécifié.
    final_dict_poids = {key: final_dict[value] for key, value in properties_dict .items()}
    del final_dict_poids['n']
    with open(file_path, 'w') as file:
        for key, value in final_dict_poids.items():
            file.write(f"{key}={value}\n")

def main():
    config_file = 'config.txt'
    if not os.path.exists(config_file):
        print(f"Fichier de configuration {config_file} introuvable.")
        return
    
    # Charger la configuration à partir du fichier config.txt
    file_path,columns_csv, properties_dict, prop_columns, num_clusters, output_file = load_config(config_file)
    data = load_data(file_path, usecols=columns_csv)
    if data is None:
        return
    
    # Prétraitement des propriétés
    data = preprocess_properties(data, properties_dict,prop_columns)
    
    # Application de K-means et calcul des poids
    list_weights = apply_kmeans(data, prop_columns, num_clusters)
    
    # Combinaison des poids
    final_weights = combine_weights(list_weights)
    
    # Écriture dans le fichier de propriétés
    write_to_properties_file(final_weights,properties_dict, output_file)
    
    # Affichage des résultats
    print(data)
    print(list_weights)
    print(final_weights)
    
  

if __name__ == "__main__":
    main()
