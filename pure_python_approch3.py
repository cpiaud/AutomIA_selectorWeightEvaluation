import re
import math
from collections import Counter
import argparse

def process_xpaths(input_filepath, output_filepath):
    # Lire les lignes depuis le fichier
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier d'entrée '{input_filepath}' n'a pas été trouvé.")
        return

    # Regex pour extraire les attributs
    attr_regex = re.compile(r'@([a-zA-Z0-9_-]+)')

    # Compter les attributs
    attribute_counter = Counter()

    for line in lines:
        attrs = attr_regex.findall(line)
        for attr in attrs:
            attribute_counter[attr] += 1

    if not attribute_counter:
        print("ℹ️ Aucun attribut trouvé dans le fichier d'entrée. Le fichier de sortie ne sera pas généré.")
        return

    # Trouver la fréquence max pour le calcul de poids
    max_count = max(attribute_counter.values())

    # Calculer les poids entre 1 et 100 avec racine carrée normalisée
    attribute_weights = {
        attr: round((math.sqrt(count) / math.sqrt(max_count)) * 100) if max_count > 0 else 0
        for attr, count in attribute_counter.items()
    }

    # Écrire dans le fichier properties
    with open(output_filepath, "w", encoding="utf-8") as f:
        for attr, weight in sorted(attribute_weights.items(), key=lambda x: -x[1]):
            f.write(f"{attr}={weight}\n")

    print(f"✅ Fichier '{output_filepath}' généré avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un fichier de poids pour les attributs XPath.")
    parser.add_argument("input_file", nargs='?', default="xpathsLists/xpath_GESICO.txt",
                        help="Chemin du fichier XPath en entrée (défaut: xpathsLists/xpath_GESICO.txt)")
    parser.add_argument("output_file", nargs='?', default="weights/selectorWeight3.properties",
                        help="Chemin du fichier properties en sortie (défaut: weights/selectorWeight3.properties)")
    args = parser.parse_args()
    process_xpaths(args.input_file, args.output_file)
