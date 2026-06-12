# Ce fichier implémente LE modèle de classification basé sur la méthode Bagging
# (Bootstrap Aggregating), utilisant un arbre de décision comme estimateur de base.
# Il contient deux fonctions principales :
#   1. train() : entraîne un modèle Bagging avec ou sans optimisation d’hyperparamètres.
#   2. evaluate() : évalue les performances du modèle sur un jeu de validation.

import pandas as pd
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train(train_df, optimise=0, model_path="models/base/bagging_model.pkl"):
    """
    Fonction d'entraînement du modèle Bagging.

    Paramètres :
    - train_df : DataFrame d'entraînement contenant les features et la cible.
    - optimise : 0 (mode simple avec hyperparamètres par défaut) ou 1 (mode optimisation avec GridSearchCV).
    - model_path : chemin de sauvegarde du modèle entraîné.
    """

    print("\nEntraînement du modèle Bagging")

    # Séparation des features et de la cible
    X_train = train_df.drop(["id_tweet", "target"], axis=1)
    y_train = train_df["target"]

    # --- Mode normal (sans optimisation)
    if not optimise:
        # Initialisation du modèle Bagging avec un arbre de décision comme estimateur de base
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, random_state=42),
            n_estimators=100,
            max_samples=0.8,
            max_features=1.0,
            bootstrap=True,
            random_state=42
        )

        # Entraînement du modèle
        model.fit(X_train, y_train)
        print("Entraînement terminé avec les hyperparamètres de base.")

    # --- Mode optimisation (GridSearchCV)
    else:
        print("Optimisation des hyperparamètres avec GridSearchCV...")

        # Estimateur de base et modèle Bagging initial
        base_estimator = DecisionTreeClassifier(random_state=42)
        bagging = BaggingClassifier(estimator=base_estimator, random_state=42)

        # Grille d’hyperparamètres à tester
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': [0.6, 0.8, 1.0],
            'max_features': [0.5, 0.8, 1.0],
            'estimator__max_depth': [4, 6, 8, None],
            'estimator__min_samples_split': [2, 5, 10],
        }

        # Configuration du GridSearchCV avec validation croisée
        grid_search = GridSearchCV(
            bagging,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        # Lancement de l'optimisation
        grid_search.fit(X_train, y_train)

        print("\nOptimisation terminée !")
        print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
        print(f"Précision moyenne (validation croisée) : {grid_search.best_score_:.4f}")

        # Le meilleur modèle est récupéré automatiquement
        model = grid_search.best_estimator_

        # Sauvegarde des résultats complets du GridSearch
        results_df = pd.DataFrame(grid_search.cv_results_)
        os.makedirs("models/opt_results", exist_ok=True)
        results_path = "models/opt_results/bagging_gridsearch_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Résultats complets sauvegardés dans {results_path}")

        # --- Visualisation 1 : évolution de la précision selon n_estimators et max_depth
        plt.figure(figsize=(8, 5))
        for depth in param_grid["estimator__max_depth"]:
            subset = results_df[results_df["param_estimator__max_depth"] == depth]
            plt.plot(subset["param_n_estimators"], subset["mean_test_score"],
                     marker="o", label=f"max_depth={depth}")
        plt.title("Évolution de la précision selon n_estimators et max_depth")
        plt.xlabel("n_estimators")
        plt.ylabel("Précision moyenne (validation croisée)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Visualisation 2 : heatmap selon max_depth et n_estimators
        heatmap_data = results_df.pivot_table(
            values="mean_test_score",
            index="param_estimator__max_depth",
            columns="param_n_estimators"
        )
        plt.figure(figsize=(7, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Précision moyenne selon max_depth et n_estimators")
        plt.xlabel("n_estimators")
        plt.ylabel("max_depth")
        plt.tight_layout()
        plt.show()

        # --- Visualisation 3 : heatmap selon max_samples et max_features
        heatmap_data2 = results_df.pivot_table(
            values="mean_test_score",
            index="param_max_samples",
            columns="param_max_features"
        )
        plt.figure(figsize=(7, 5))
        sns.heatmap(heatmap_data2, annot=True, fmt=".3f", cmap="coolwarm")
        plt.title("Précision moyenne selon max_samples et max_features")
        plt.xlabel("max_features")
        plt.ylabel("max_samples")
        plt.tight_layout()
        plt.show()

    # --- Sauvegarde du modèle final
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    return model


def evaluate(val_df, model_path="models/base/bagging_model.pkl"):
    """
    Fonction d'évaluation du modèle Bagging sauvegardé.

    Paramètres :
    - val_df : DataFrame contenant les données de validation.
    - model_path : chemin du fichier .pkl du modèle sauvegardé.
    """

    print("\nÉvaluation du modèle Bagging")

    # Charger le modèle
    model = joblib.load(model_path)

    # Préparation des features et de la cible
    X_val = val_df.drop(["id_tweet", "target"], axis=1)
    y_val = val_df["target"]

    # Prédictions
    y_pred = model.predict(X_val)

    # Évaluation
    acc = accuracy_score(y_val, y_pred)
    print(f"Exactitude (Accuracy) : {acc:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Matrice de confusion - Bagging")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()
