# AutomIA_selectorWeightEvaluation
Use Machine Learning to evaluate Weight of properties to find Element with AutomIA_ElementFinder.

Ce travail consite à déterminer les poids de chaque propriété en utilisant les méthodes de machine learning(clustring) et de l'IA(OpenAI) :
- Le fichier en sortie de machine learning doit s'appeller "selectorWeight.properties" et être constitué d'une liste de pair clé:valeur.
- La Clé est un attribut/propriété d'un élément (ex: id, class, name, type...)
- La valeur est un poids de 0 à 100 qui donne l'importance de l'attribut pour retrouver un élément par rapport aux autres attributs.

## Approche 01: Clustring avec K-means

K-means partitionne 𝑛 observations en 𝑘 clusters en assignant chaque point au centre de cluster le plus proche. Les centres de clusters sont recalculés comme la moyenne des points assignés, et le processus est répété jusqu'à convergence. L'algorithme minimise la variance intra-cluster pour une classification optimale.
Pour utiliser cette méthode , on va suivre les étapes suivantes:
### 1- Préparation des données
La préparation des données consiste à structurer nos données sous forme CSV et remplir les cases vides par "n"(no data) .
Voir un exemple de préparation sous forme data.csv
### 2- Configuration des entrées/sorties
Configuration des parametres dans un fichier config_kmeans.txt sous la forme:

[base]

file_path=

output_file=selectorWeight_data.properties

num_clusters=

[columns_csv]

cols=

[properties_dict]

[prop_columns]

prop_columns = 

- [base] : comprend le path des data, le nom de l'output et le num_clusters (nombre des propriétés unique +1).
- [columns_csv] : comprend tous les noms des colonnes dans le fichier csv.
- [properties_dict] : attribuer à chaque propriété(id,class,name,etc) un numéro (entre 1 et nombre de propriétés et 0 pour "n").
- [prop_columns] : la liste des colonnes qui sont déstinés pour les propriétés .

Voir un exemple de fichier : config.txt

## 3-Installation et exécution 
- Créer un environement virtuel :
```bash
Python -m venv myenv
```

- Créer un environement virtuel :
```bash
pip install -r requirements.txt
```

- Exécuter le code:
```bash
python kmeans_clustring_approch1.py
```

## Approche 02: Calculer les poids avec OpenAI
L'API d'OpenAI permet d'intégrer des capacités avancées d'intelligence artificielle dans les applications. Elle offre des fonctionnalités telles que la génération de texte, la compréhension du langage, et la création de réponses intelligentes, basées sur des modèles de traitement du langage naturel comme GPT-4. Les utilisateurs peuvent envoyer des requêtes et recevoir des réponses adaptées à des besoins variés, allant de la création de contenu à l'assistance client. Pour accéder à l'API, une clé d'API est nécessaire et l'utilisation est généralement facturée en fonction du volume de requêtes.
Pour utiliser cette méthode , on va suivre les étapes suivantes:
### 1-Géneration d'un token OpenAI
- Créer un compte sur : https://platform.openai.com/
- Créer un projet et configurez les options de facturation.
- Générer la clé API et copiez-la dans un fichier sécurisé api_key_openai.txt sous configs.
### 2- Préparation des données(meme que l'approche 1)
La préparation des données consiste à structurer nos données sous forme CSV et remplir les cases vides par "n"(no data) .
Voir un exemple de préparation sous forme data.csv
### 3- Configuration des entrées/sorties
Configuration des parametres dans un fichier config_gpt.txt sous la forme:

[base]
file_path=data.csv
output_file=selectorWeight_data_gpt.properties

[columns_csv]
cols=ElementName, Langage, Tag, Prop1, Prop2, Prop3, Prop4, Prop5

Voir un exemple de fichier : config_gpt.txt.

Lister dans le code open_ai_approch2.py dans la section messages les propriétés:
```bash
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
```

## 4-Installation et exécution 
- Créer un environement virtuel :
```bash
Python -m venv myenv
```

- Créer un environement virtuel :
```bash
pip install -r requirements.txt
```

- Exécuter le code:
```bash
python open_ai_approch2.py
```
## Resultats
Les résultats obtenus avec les deux approches sont comparables en ce qui concerne la force des poids obtenus. 
NB : les valeurs des poids ne sont pas identiques pour les deux approches, mais elles peuvent être expliquées de la même manière.
