# Ce fichier implémente le modèle de classification basé sur la méthode
# Gradient Boosting (GradientBoostingClassifier).
#
# Il contient deux fonctions principales :
#   1. train() : entraîne un modèle avec ou sans optimisation d’hyperparamètres.
#   2. evaluate() : évalue les performances du modèle à l’aide de métriques
#                   classiques (accuracy, classification_report, confusion matrix).

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train(train_df, optimise=0, model_path="models/base/gboost_model.pkl"):
    """
    Fonction d'entraînement du modèle Gradient Boosting.

    Paramètres :
    - train_df : DataFrame contenant les données d'entraînement (features + cible).
    - optimise : 0 = entraînement classique, 1 = optimisation via GridSearchCV.
    - model_path : chemin où sauvegarder le modèle entraîné (.pkl).
    """

    print("Entraînement du modèle Gradient Boosting...")

    # Séparation des variables explicatives et de la cible
    X_train = train_df.drop(["id_tweet", "target"], axis=1)
    y_train = train_df["target"]

    if optimise == 0:
        # Modèle avec paramètres définis manuellement
        model = GradientBoostingClassifier(
            n_estimators=200,      # Nombre d'arbres (itérations)
            learning_rate=0.1,     # Taux d'apprentissage (impact de chaque arbre)
            max_depth=3,           # Profondeur maximale des arbres
            subsample=0.8,         # Fraction d'échantillons pour chaque arbre
            random_state=42
        )
        model.fit(X_train, y_train)
        print("Entraînement terminé.")

    else:
        if optimise == 1:
            print("Optimisation des hyperparamètres avec GridSearchCV...")

            # Grille de recherche des paramètres
            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.6, 0.8, 1.0]
            }

            base_model = GradientBoostingClassifier(random_state=42)

            # Configuration du GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=2
            )

            # Lancement de l'optimisation
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            print("Meilleurs paramètres trouvés :", grid_search.best_params_)
            print("Meilleure précision obtenue :", grid_search.best_score_)

            # Sauvegarde des résultats complets
            results_df = pd.DataFrame(grid_search.cv_results_)
            os.makedirs("models/opt_results", exist_ok=True)
            results_path = "models/opt_results/gboost_gridsearch_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"Résultats complets sauvegardés dans {results_path}")

            # Visualisation : précision moyenne selon n_estimators et learning_rate
            plt.figure(figsize=(8, 5))
            for lr in param_grid["learning_rate"]:
                subset = results_df[results_df["param_learning_rate"] == lr]
                plt.plot(subset["param_n_estimators"], subset["mean_test_score"], marker="o", label=f"learning_rate={lr}")
            plt.title("Précision moyenne (CV) selon n_estimators et learning_rate")
            plt.xlabel("n_estimators")
            plt.ylabel("Précision moyenne (validation croisée)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Heatmap : impact conjoint max_depth et n_estimators
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

    # Sauvegarde du modèle final
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    return model

def evaluate(val_df, model_path="models/base/gboost_model.pkl"):
    """
    Fonction d'évaluation d'un modèle Gradient Boosting sauvegardé.

    Paramètres :
    - val_df : DataFrame de validation.
    - model_path : chemin du fichier modèle sauvegardé.
    """

    print("\nÉvaluation du modèle Gradient Boosting...")

    # Chargement du modèle
    model = joblib.load(model_path)

    # Séparation X/y sur les données de validation
    X_val = val_df.drop(["id_tweet", "target"], axis=1)
    y_val = val_df["target"]

    # Prédictions
    y_pred = model.predict(X_val)

    # Évaluation globale (accuracy + classification report)
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title("Matrice de confusion - Gradient Boosting")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()
