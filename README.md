# AutomIA_selectorWeightEvaluation
Use Machine Learning to evaluate Weight of properties to find Element with AutomIA_ElementFinder.
Ce travail consite à déterminer les poids de chaque propriété en utilisant les méthodes de machine learning(clustring) et de l'IA(OpenAI) :
- Le fichier en sortie de machine learning doit s'appeller "selectorWeight.properties" et être constitué d'une liste de pair clé:valeur.
- La Clé est un attribut/propriété d'un élément (ex: id, class, name, type...)
- La valeur est un poids de 0 à 100 qui donne l'importance de l'attribut pour retrouver un élément par rapport aux autres attributs.

## Clustring avec K-means
K-means partitionne 𝑛 observations en 𝑘 clusters en assignant chaque point au centre de cluster le plus proche. Les centres de clusters sont recalculés comme la moyenne des points assignés, et le processus est répété jusqu'à convergence. L'algorithme minimise la variance intra-cluster pour une classification optimale.
Pour utiliser cette méthode , on va suivre les étapes suivantes:
### 1- Préparation des données
La préparation des données consiste à structurer nos données sous forme CSV et remplir les cases vides par "n"(no data) .

Voir un exemple de préparation sous forme data.csv

### 2- Configuration des entrées/sorties
Configuration des parametres dans un fichier config.txt sous la forme:

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
- Créer et activer un environement virtuel :
```bash
Python -m venv myenv
venv\Scripts\activate
```

- Créer un environement virtuel :
```bash
pip install -r requirements.txt
```

- Exécuter le code:
```bash
python kmeans_clustring_approch1.py
```
## Done!
## ToDO
- Proposer Autres approches de Machine Learning.
- Utiliser Openai comme une solution.
- Comparer les résultats et déployer la solution.
