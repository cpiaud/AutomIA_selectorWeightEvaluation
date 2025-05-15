# AutomIA_selectorWeightEvaluation Version 02
Use Machine Learning to evaluate Weight of properties to find Element with AutomIA_ElementFinder.

Ce travail consite à déterminer les poids de chaque propriété en utilisant les méthodes de machine learning(clustring) et de l'IA(OpenAI) :
- Le fichier en sortie de machine learning doit s'appeller "selectorWeight.properties" et être constitué d'une liste de pair clé=valeur.
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
```bash
[base]

file_path=

output_file=selectorWeight_data_kmeans_approch1.properties

num_clusters=

[columns_csv]

cols=

[properties_dict]

[prop_columns]

prop_columns = 
```

- [base] : comprend le path des data(csv), le nom de l'output et le num_clusters (nombre des propriétés unique +1).
- [columns_csv] : comprend tous les noms des colonnes dans le fichier csv.
- [properties_dict] : attribuer à chaque propriété(id,class,name,etc) un numéro (entre 1 et nombre de propriétés et 0 pour "n").
- [prop_columns] : la liste des colonnes qui sont déstinés pour les propriétés .

Voir un exemple de fichier : config_kmeans.txt

## 3-Installation et exécution 
- Créer un environement virtuel :
```bash
Python -m venv myenv
```

- Installer les packages nécessaires :
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
```bash
[base]

file_path=

output_file=selectorWeight_data_gpt.properties

[columns_csv]

cols=
```
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

### 4-Installation et exécution 
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
### 5-Resultats
Les résultats obtenus avec les deux approches(voir weights/...) sont comparables en ce qui concerne la force des poids obtenus. 

NB : les valeurs des poids ne sont pas identiques pour les deux approches, mais elles peuvent être expliquées de la même manière.Voci un exemple des résultats pour les données utilisés (data.csv):


| propriété       | Kmeans | openai |
|----------------|---------|---------|
| data-focus     | 5       | 10      |
| label          | 15      | 30      |
| id             | 24      | 50      |
| profil-list    | 5       | 10      |
| aria-label     | 19      | 30      |
| class          | 29      | 60      |
| text           | 5       | 10      |
| name           | 5       | 10      |
| for            | 5       | 10      |
| grid           | 5       | 10      |
| index          | 19      | 40      |


Les approches sont généralisées , il suffit de suivre les étapes et les configurations nécessaires.

## Approach 03: Calculate weights in pure Python based on property representativeness
This approach uses the pure_python_approch3.py script to calculate attribute weights based on their frequency of appearance in a list of XPaths.

### 1-Principle
The script pure_python_approch3.py performs the following steps:

- Reads a list of XPaths from a specified input file. Each line in the file is expected to be a single XPath.
- Parses each XPath to find all attributes (e.g., @id, @class, @name).
- Counts the occurrences of each unique attribute across all XPaths.
- Calculates a weight for each attribute. The weight is determined by its frequency:
    - The square root of the attribute's count is taken.
    - This value is then divided by the square root of the maximum count found for any attribute (to normalize it).
    - The result is multiplied by 100 and rounded to the nearest integer, yielding a weight between 0 and 100.
    - Attributes that appear more frequently will receive higher weights.
- Writes the calculated attribute weights to a specified output .properties file, with each line in the format attribute=weight.

### 2-How to Use
**Prerequisites**
- Python 3 environment.
- The script uses standard Python libraries (re, math, collections, argparse), so no special installation of external packages is required beyond a standard Python installation.
**Execution**

The script is run from the command line. It accepts two optional positional arguments:

- input_file: (Optional) The path to the input file containing XPaths, one XPath per line.
    - Default value: xpathsLists/xpath_GESICO.txt
- output_file: (Optional) The path to the output .properties file where the attribute weights will be saved.
    - Default value: weights/selectorWeight3.properties
**Examples**

- Using default input and output paths:
```bash
python pure_python_approch3.py
```
- Specifying only the input file (output will use default):
```bash
python pure_python_approch3.py my_xpath_list.txt
```
- Specifying both input and output files:
```bash
python pure_python_approch3.py path/to/your/xpaths.txt path/to/your/output_weights.properties
```
- Getting help on arguments:
```bash
python pure_python_approch3.py -h
```
**Output**

The script generates a .properties file (e.g., selectorWeight3.properties) containing key-value pairs, where the key is the attribute name and the value is its calculated weight (0-100). The attributes are sorted by weight in descending order in the output file.