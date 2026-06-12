# Ce fichier implémente le modèle de classification basé sur un Arbre de Décision
# (Decision Tree). Il propose deux fonctionnalités principales :
#   1. train() : entraîner un modèle avec ou sans optimisation d’hyperparamètres.
#   2. evaluate() : évaluer les performances du modèle à l’aide de métriques et d’une matrice de confusion.

import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train(train_df, optimise=0, model_path="models/base/decision_tree_model.pkl"):
    """
    Fonction d'entraînement d'un modèle Arbre de Décision.

    Paramètres :
    - train_df : DataFrame contenant les données d'entraînement (features + cible).
    - optimise : 0 = entraînement classique avec hyperparamètres définis.
                 1 = optimisation via GridSearchCV (recherche des meilleurs paramètres).
    - model_path : chemin où sauvegarder le modèle entraîné (.pkl).
    """

    print("Entraînement du modèle Arbre de Décision...")

    # Séparer features et target
    X_train = train_df.drop(["id_tweet", "target"], axis=1)
    y_train = train_df["target"]

    if optimise == 0:
        # Modèle avec paramètres par défaut / arbitraires choisis
        model = DecisionTreeClassifier(
            criterion="gini",           # Mesure d'impureté
            max_depth=5,                # Profondeur maximale de l'arbre
            min_samples_split=10,       # Nombre min. d'échantillons pour diviser un nœud
            min_samples_leaf=5,         # Nombre min. d'échantillons par feuille
            random_state=42
        )
        model.fit(X_train, y_train)
        print("Entraînement terminé.")

    else:
        if optimise == 1:
            print("Optimisation des hyperparamètres avec GridSearchCV...")

            # Grille d'hyperparamètres à tester
            param_grid = {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "criterion": ["gini", "entropy"]
            }

            base_model = DecisionTreeClassifier(random_state=42)

            # Lancement de la recherche en grille
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=2
            )

            # Ajustement du modèle avec validation croisée
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            print("Meilleurs paramètres trouvés :", grid_search.best_params_)
            print("Meilleure précision obtenue :", grid_search.best_score_)

            # Sauvegarder les résultats complets du GridSearch
            results_df = pd.DataFrame(grid_search.cv_results_)
            os.makedirs("models/opt_results", exist_ok=True)
            results_path = "models/opt_results/decision_tree_gridsearch_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"Résultats complets sauvegardés dans {results_path}")

            # Visualisation : évolution de la précision selon max_depth et criterion
            plt.figure(figsize=(8, 5))
            for crit in param_grid["criterion"]:
                subset = results_df[results_df["param_criterion"] == crit]
                plt.plot(subset["param_max_depth"], subset["mean_test_score"], marker="o", label=f"criterion={crit}")
            plt.title("Précision moyenne (CV) selon max_depth et criterion")
            plt.xlabel("max_depth")
            plt.ylabel("Précision moyenne (validation croisée)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Heatmap : mean_test_score selon max_depth et min_samples_split
            heatmap_data = results_df.pivot_table(
                values="mean_test_score",
                index="param_max_depth",
                columns="param_min_samples_split"
            )
            plt.figure(figsize=(7, 5))
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
            plt.title("Précision moyenne selon max_depth et min_samples_split")
            plt.xlabel("min_samples_split")
            plt.ylabel("max_depth")
            plt.tight_layout()
            plt.show()

    # Sauvegarde du modèle final au format pickle
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    return model

def evaluate(val_df, model_path="models/base/decision_tree_model.pkl"):
    """
    Fonction d'évaluation du modèle Arbre de Décision.

    Paramètres :
    - val_df : DataFrame contenant les données de validation.
    - model_path : chemin du fichier modèle sauvegardé.
    """

    print("Évaluation du modèle Arbre de Décision...")
    model = joblib.load(model_path)

    # Séparation du dataset en features et cible
    X_val = val_df.drop(["id_tweet", "target"], axis=1)
    y_val = val_df["target"]

    # Prédictions
    y_pred = model.predict(X_val)

    # Évaluation des performances
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Rapport de classification :")
    print(classification_report(y_val, y_pred))

    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion - Arbre de Décision")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()

    return acc
