import openai
import pandas as pd
import configparser
import os
from utils import load_config,load_data
# Lire la clé API depuis un fichier
def read_api_key(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Lire la clé API depuis le fichier
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_api_key = os.path.join(script_dir, 'configs', 'api_key_openai.txt')
api_key = read_api_key(config_file_api_key)

# Configurer OpenAI avec la clé API lue
openai.api_key = api_key

def get_response_gpt(data):
    # Préparer le message pour le chat
    messages = [
        {
            "role": "system",
            "content": "Vous êtes un assistant chargé de calculer les poids des propriétés des éléments en fonction des données fournies."
        },
        {
            "role": "user",
            "content": f"""
            Voici des données sur des éléments en Angular JS et Vue JS avec leurs propriétés :
        {data}

        Veuillez calculer et afficher le poids total de chaque propriété mentionnée ci-dessous en fonction de leur apparition dans les données :

        - data-focus
        - label
        - id
        - profil-list
        - aria-label
        - class
        - text
        - name
        - for
        - grid
        - index

        Pour chaque propriété, affichez le poids total associé.
        """
        }
        ]

        # Utiliser l'API OpenAI pour obtenir la réponse en mode chat
    response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500
    )

    return  response.choices[0].message.content.strip()

def save_weights(response,output_file):
    lines = response.split('\n')

    # Filtrer, formater et multiplier les poids
    formatted_lines = []
    for line in lines:
        if line.startswith("-"):
            # Enlever les tirets et extraire la propriété et le poids
            formatted_line = line.replace("- ", "").strip()
            property_name, weight_str = formatted_line.split(': ')
            weight = int(weight_str) * 10
            formatted_lines.append(f"{property_name}={weight}")

    # Afficher les résultats formatés
    formatted_result = "\n".join(formatted_lines)
    print(formatted_result)

    # Si vous souhaitez également écrire les résultats dans un fichier texte
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(formatted_result)
def main():
    #config_file=os.path.join(script_dir, 'configs', )

    file_path,output_file,columns_csv=load_config(script_dir,'config_gpt.txt')
    data=load_data(file_path,columns_csv)
    response=get_response_gpt(data)
    output_file1 = os.path.join(script_dir,'weights', output_file)
    print(output_file1)
    save_weights(response,output_file1)
    
if __name__ == "__main__":
    main()
    

