# Ce fichier est dédié à l'entraînement et à l'évaluation du modèle de Random Forest
# Il inclut deux fonctionnalités principales :
#   1. train() : entraîner le modèle avec ou sans optimisation des hyperparamètres.
#   2. evaluate() : évaluer les performances du modèle sur un jeu de validation.

import pandas as pd 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train(train_df, optimise=0, model_path="models/base/random_forest_model.pkl"):
    """
    Fonction d'entraînement du modèle Random Forest.
    
    Paramètres :
    - train_df : DataFrame contenant les features et la variable cible.
    - optimise : 0 (entraînement simple) ou 1 (optimisation des hyperparamètres via GridSearchCV).
    - model_path : chemin de sauvegarde du modèle entraîné.
    """
    print("Entraînement du modèle Random Forest...")

    # Séparation des features et de la variable cible
    X_train = train_df.drop(["id_tweet", "target"], axis=1)
    y_train = train_df["target"]

    if optimise == 0:
        # Initialisation du modèle avec des paramètres définis manuellement
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)
        print("Entraînement terminé.")
    
    else :
        if optimise == 1:
            print("Optimisation des hyperparamètres avec GridSearchCV...")

            # Définition de la grille de recherche pour les hyperparamètres
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False]
            }

            # Modèle de base pour la recherche
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

            # Configuration et exécution du GridSearchCV (validation croisée)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            print("Meilleurs paramètres trouvés :", grid_search.best_params_)
            print("Meilleure précision obtenue :", grid_search.best_score_)

            # Sauvegarder les résultats complets du GridSearch
            results_df = pd.DataFrame(grid_search.cv_results_)
            os.makedirs("models/opt_results", exist_ok=True)
            results_path = "models/opt_results/random_forest_gridsearch_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"Résultats complets sauvegardés dans {results_path}")

            # Visualisation 1 : précision moyenne selon n_estimators et max_depth
            plt.figure(figsize=(8, 5))
            for depth in param_grid["max_depth"]:
                subset = results_df[results_df["param_max_depth"] == depth]
                plt.plot(subset["param_n_estimators"], subset["mean_test_score"], marker="o", label=f"max_depth={depth}")

            plt.title("Évolution de la précision selon n_estimators et max_depth")
            plt.xlabel("n_estimators")
            plt.ylabel("Précision moyenne (validation croisée)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Visualisation 2 : heatmap selon deux paramètres
            heatmap_data = results_df.pivot_table(
                values="mean_test_score",
                index="param_max_depth",
                columns="param_n_estimators"
            )

            plt.figure(figsize=(7, 5))
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
            plt.title("Précision moyenne selon max_depth et n_estimators")
            plt.xlabel("n_estimators")
            plt.ylabel("max_depth")
            plt.tight_layout()
            plt.show()

    # Sauvegarde du modèle
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    return model


def evaluate(val_df, model_path="models/base/random_forest_model.pkl"):
    """
    Paramètres :
    - val_df : DataFrame de validation contenant les features et la cible.
    - model_path : chemin vers le modèle enregistré à tester.
    """
    print("\nÉvaluation du modèle Random Forest...")

    # Chargement du modèle entraîné
    model = joblib.load(model_path)

    # Séparation des features et de la cible pour l'évaluation
    X_val = val_df.drop(["id_tweet", "target"], axis=1)
    y_val = val_df["target"]

    # Prédictions
    y_pred = model.predict(X_val)

    # Évaluation des performances
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))

    # Matrice de confusion (visualisation)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Matrice de confusion - Random Forest")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()
