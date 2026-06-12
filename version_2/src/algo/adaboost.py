import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def train(train_df, optimise=0, model_path="models/base/adaboost_model.pkl"):
    print("Entraînement du modèle AdaBoost...")

    X_train = train_df.drop(["id_tweet", "target"], axis=1)
    y_train = train_df["target"]

    if optimise == 0:
        model = AdaBoostClassifier(
            n_estimators=500,
            learning_rate=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("Entraînement terminé.")

    elif optimise == 1:
        print("Optimisation des hyperparamètres avec GridSearchCV...")

        param_grid = {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.1, 0.5, 0.8, 1.0]
        }

        base_model = AdaBoostClassifier(random_state=42)

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

        # Sauvegarder les résultats complets
        results_df = pd.DataFrame(grid_search.cv_results_)
        os.makedirs("models/opt_results", exist_ok=True)
        results_path = "models/opt_results/adaboost_gridsearch_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Résultats complets sauvegardés dans {results_path}")

        # Visualisation 1 : évolution de la précision selon n_estimators
        plt.figure(figsize=(8, 5))
        for lr in param_grid["learning_rate"]:
            subset = results_df[results_df["param_learning_rate"] == lr]
            plt.plot(subset["param_n_estimators"], subset["mean_test_score"], marker="o", label=f"learning_rate={lr}")

        plt.title("Évolution de la précision selon n_estimators et learning_rate")
        plt.xlabel("n_estimators")
        plt.ylabel("Précision moyenne (validation croisée)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Visualisation 2 : heatmap précision selon deux paramètres
        heatmap_data = results_df.pivot_table(
            values="mean_test_score",
            index="param_learning_rate",
            columns="param_n_estimators"
        )

        plt.figure(figsize=(7, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Précision moyenne selon learning_rate et n_estimators")
        plt.xlabel("n_estimators")
        plt.ylabel("learning_rate")
        plt.tight_layout()
        plt.show()

    # Sauvegarde du modèle final
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    return model

def evaluate(val_df, model_path="models/base/adaboost_model.pkl"):
    print("Évaluation du modèle AdaBoost...")
    model = joblib.load(model_path)

    X_val = val_df.drop(["id_tweet", "target"], axis=1)
    y_val = val_df["target"]

    # Prédictions
    y_pred = model.predict(X_val)

    # Évaluation
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))

    # Matrice de confusion (visualisation)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion - AdaBoost")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()