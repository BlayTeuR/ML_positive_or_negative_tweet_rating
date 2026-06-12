## JallaisBastien_INF7370_TP1

# TP1 - INF7370 - Apprentissage Automatique

**IMPORTANT** : Les détails concernant le choix des caractéristiques, le prétraitement des données, ainsi que l'analyse comparative des algorithmes sont disponibles dans le fichier **rapport_Jallais_Bastien_TP1**.

Le projet a pour but d'effectuer une classification de tweets selon leur connotation positive ou négative. Pour cela, plusieurs algorithmes sont utilisés afin de tester leur fonctionnement et d'évaluer quantitativement les résultats.

## Architecture du projet

- Le code du projet est disponible dans le dossier "src". Chaque algorithme est écrit dans un fichier différent contenu dans le dossier "src/algo". Le code de chaque algo contient 2 fonctions : une servant à l’entraîner, l’autre servant à l’évaluer.
- Le dossier Backup contient les fichiers train.csv ou test.csv afin de pouvoir relancer aisément le projet si le fichier training_data.csv vient à être corrompu.

-> Pour chaque fichier, des commentaires sont disponibles pour comprendre plus en détail le code mis en place.

## Exécution du projet

Afin de faciliter la prise en main et la compréhension du projet, un petit menu permettant d’entraîner et d’évaluer des modèles a été mis en place. Pour lancer celui-ci, il suffit de faire "python training.py" dans le dossier /src.

La construction des fichiers "test_data.csv" ainsi que "test_prediction.csv" se fait en lançant "python test.py". La construction de ses fichiers n'est pas possible via le menu car l'explication du choix du modèle ainsi que l'analyse des résultats obtenus est déjà disponible dans le rapport. 

Le menu se découpe en 4 grandes parties :

### 1) Chargement des données d’entraînement et du DataFrame correspondant (choix 1).

Dans cette partie, si le fichier training_data.csv n’existe pas, il sera automatiquement créé et les DataFrames chargés en mémoire. En revanche, si celui-ci existe, deux options sont possibles : charger le DataFrame à partir du fichier existant (ATTENTION : ESSENTIEL DE LE FAIRE AU MOINS UNE FOIS AVANT DE TENTER D’ENTRAÎNER OU D'EVALUER DES MODÈLES), ou alors recréer le fichier de zéro.

### 2) Entraînement des modèles (choix 2)

Il est possible d’entraîner deux types de modèles différents pour chaque algorithme utilisé dans ce projet : soit un modèle avec des hyperparamètres classiques par défaut, soit un modèle utilisant l’outil GridSearchCV pour tenter de maximiser les performances du modèle en cherchant les meilleurs hyperparamètres. Attention, certains algorithmes comme Bagging ou RandomForest peuvent être assez longs à entraîner avec GridSearchCV.

Pour faciliter l’évaluation des algorithmes, des versions de base et optimisées (GridSearchCV) de chaque modèle sont déjà disponibles et prêtes à être utilisées sur des données de test. Les versions basiques des modèles sont dans le dossier "/src/models/base" et les versions optimisées dans "/src/models/opt_results".

### 3) Évaluation des modèles (choix 3)

Tout comme pour l’entraînement, il est possible d’évaluer les modèles de base ainsi que les modèles optimisés. Chaque évaluation d’un modèle donne son exactitude (accuracy) ainsi que la matrice de confusion correspondante (les calculs de précision, rappel et f-mesure sont disponibles dans le rapport pour chaque modèle).

ATTENTION -> Il est nécessaire d’entraîner des modèles avant de les évaluer.

### 4) Comparaison des modèles (choix 3)

Cette fonctionnalité génère un graphe de comparaison montrant les différences d'éfficacité entre les différents modèles selon les critères comme l'accuracy, le rappel, la précisio ou encore f_mesure.

ATTENTION -> Il est nécessaire d’entraîner des modèles avant de les comparer.

## Prérequis pour exécuter le projet :

- scikit-learn
- pandas
- matplotlib
- seaborn
