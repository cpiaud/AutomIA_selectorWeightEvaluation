#import packages
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter, OrderedDict
import configparser
import os
from utils import load_config,load_data
#current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
def preprocess_properties(data, properties_dict, prop_columns):
    """ 
    Transforme les propriétés dans un DataFrame en utilisant le dictionnaire de propriétés fourni.

    Arguments:
    data -- DataFrame contenant les données à transformer. 
    properties_dict -- Dictionnaire où les clés sont les valeurs actuelles des propriétés et les valeurs sont les nouvelles valeurs désirées.
    prop_columns -- Liste des noms des colonnes dans le DataFrame 'data' qui contiennent les propriétés à transformer.

    Retourne:
    DataFrame -- Le DataFrame 'data' avec les propriétés transformées selon 'properties_dict'.
    """
    for prop in prop_columns:
        data[prop] = data[prop].map(properties_dict)
    return data

def apply_kmeans(data, prop_columns, num_clusters):
    """
    Applique l'algorithme K-means sur des colonnes spécifiques du DataFrame et calcule les poids des clusters.

    Arguments:
    data -- DataFrame contenant les données à analyser. Les colonnes spécifiées doivent être numériques.
    prop_columns -- Liste des noms des colonnes sur lesquelles appliquer K-means.
    num_clusters -- Nombre de clusters à créer avec l'algorithme K-means.

    Retourne:
    list_dict_weights -- Liste de dictionnaires, où chaque dictionnaire contient les poids des clusters 
                         pour une colonne spécifique.
    """
    # Initialise les centres de clusters comme une liste de listes contenant des indices allant de 0 à num_clusters-1.
    initial_centers = [[i] for i in range(num_clusters)]
    
    # Liste pour stocker les poids des clusters pour chaque colonne traitée.
    list_dict_weights = []

    # Itère à travers chaque colonne spécifiée pour appliquer K-means.
    for prop in prop_columns:
        # Création et configuration du modèle KMeans.
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1, init=initial_centers)
        
        # Application de K-means sur la colonne actuelle et ajout des labels de clusters au DataFrame.
        data[f'cluster_{prop}'] = kmeans.fit_predict(data[[prop]])
        
        # Comptage du nombre d'échantillons dans chaque cluster.
        cluster_counts = Counter(data[f'cluster_{prop}'])
        
        # Calcul des poids des clusters en pourcentage.
        total = sum(cluster_counts.values())  # Nombre total d'échantillons
        weights = {k: round(v / total * 100) for k, v in cluster_counts.items()}  # Poids en pourcentage
        
        # Ajout des poids des clusters pour cette colonne à la liste des poids.
        list_dict_weights.append(weights)
    
    # Retourne la liste des dictionnaires de poids des clusters pour chaque colonne traitée.
    return list_dict_weights


def combine_weights(list_weights):
    """
    Combine et trie les poids des clusters de plusieurs dictionnaires en un seul dictionnaire trié.

    Arguments:
    list_weights -- Liste de dictionnaires où chaque dictionnaire contient les poids des clusters pour une colonne donnée.

    Retourne:
    sorted_weights -- Dictionnaire ordonné avec les poids combinés et triés par clé.
    """
    # Combine les poids de tous les dictionnaires dans list_weights en un seul dictionnaire.
    # Utilise Counter pour additionner les valeurs associées aux mêmes clés dans chaque dictionnaire.
    combined_weights = sum((Counter(d) for d in list_weights), Counter())
    
    # Trie les poids combinés par clé et crée un OrderedDict pour maintenir l'ordre.
    sorted_weights = OrderedDict(sorted(combined_weights.items()))
    
    # Retourne le dictionnaire trié avec les poids combinés.
    return sorted_weights


def write_to_properties_file(final_dict, properties_dict, file_path):
    """
    Écrit le dictionnaire final dans un fichier de propriétés au format clé=valeur.

    Arguments:
    final_dict -- Dictionnaire contenant les poids finaux à écrire dans le fichier.
    properties_dict -- Dictionnaire qui mappe les clés du fichier de propriétés aux clés du dictionnaire final.
    file_path -- Chemin du fichier où les propriétés doivent être écrites.

    Aucun retour.
    """
    # Prépare un dictionnaire pour le fichier de propriétés en mappant les clés de properties_dict 
    # aux valeurs correspondantes dans final_dict.
    final_dict_poids = {key: final_dict[value] for key, value in properties_dict.items()}
    
    # Supprime l'entrée avec la clé 'n' si elle existe dans le dictionnaire préparé.
    del final_dict_poids['n']
    
    # Ouvre le fichier à l'emplacement spécifié en mode écriture.
    with open(file_path, 'w') as file:
        # Écrit chaque clé et valeur dans le fichier sous la forme 'clé=valeur'.
        for key, value in final_dict_poids.items():
            file.write(f"{key}={value}\n")


def main():

    # Charger la configuration à partir du fichier config.txt
    file_path,columns_csv, properties_dict, prop_columns, num_clusters, output_file = load_config(script_dir,'config_kmeans.txt')
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
    write_to_properties_file(final_weights,properties_dict, os.path.join(script_dir+'/weights', output_file))
    
    # Affichage des résultats
    print(data)
    print(list_weights)
    print(final_weights)
    
  

if __name__ == "__main__":
    main()
