# training_menu.py
from pre_traitement_donnees import *
from data_preparation import *
from algo import adaboost, arbre_de_decision, bagging, gboost, random_forest
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

MODEL_PATHS = {
    "AdaBoost": {
        "Base": "models/base/adaboost_model.pkl",
        "Optimisé": "models/opt_results/adaboost_optimise_model.pkl"
    },
    "Bagging": {
        "Base": "models/base/bagging_model.pkl",
        "Optimisé": "models/opt_results/bagging_optimise_model.pkl"
    },
    "Arbre de Décision": {
        "Base": "models/base/decision_tree_model.pkl",
        "Optimisé": "models/opt_results/arbre_de_décision_optimise_model.pkl"
    },
    "GBoost": {
        "Base": "models/base/gboost_model.pkl",
        "Optimisé": "models/opt_results/gboost_optimise_model.pkl"
    },
    "Random Forest": {
        "Base": "models/base/random_forest_model.pkl",
        "Optimisé": "models/opt_results/random_forest_optimise_model.pkl"
    }
}

def comparer_modeles(df_val):
    """Compare les modèles existants (base + optimisés) si disponibles."""
    
    X_val = df_val.drop(["id_tweet", "target"], axis=1)
    y_val = df_val["target"]

    results = []
    print("\n--- Comparaison des modèles (Base + Optimisés) ---")

    for model_name, versions in MODEL_PATHS.items():
        for version_label, path in versions.items():
            if os.path.exists(path):
                print(f"Chargement : {model_name} ({version_label})")
                model = joblib.load(path)

                y_pred = model.predict(X_val)
                report = classification_report(y_val, y_pred, output_dict=True)

                results.append({
                    "Modèle": f"{model_name} ({version_label})",
                    "Accuracy": report["accuracy"],
                    "Précision": report["macro avg"]["precision"],
                    "Rappel": report["macro avg"]["recall"],
                    "F1-score": report["macro avg"]["f1-score"]
                })
            else:
                print(f"❌ {model_name} ({version_label}) introuvable → À entraîner avant comparaison.")

    if not results:
        print("\nAucun modèle n'a pu être comparé. Veuillez en entraîner au moins un.")
        return

    df_results = pd.DataFrame(results)
    print("\nRésultats comparatifs :")
    print(df_results)

    # Graphique comparatif
    results_melted = df_results.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_melted, x="Modèle", y="Valeur", hue="Métrique")
    plt.title("Comparaison des performances (Base vs Optimisé)")
    plt.xticks(rotation=30)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # Conclusion automatique
    best = df_results.sort_values(by="F1-score", ascending=False).iloc[0]
    print(f"\nMeilleur modèle : {best['Modèle']} avec F1-score = {best['F1-score']:.4f}")

# Variables globales pour stocker le DataFrame
df = None
train_df = None
val_df = None

def charger_training_data():
    """Charge le CSV en mémoire et sépare en train/val"""
    global df, train_df, val_df
    df = pd.read_csv("../training_data.csv", encoding="ISO-8859-1")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training et validation DataFrames prêts ({len(train_df)} / {len(val_df)})")

def construire_training_data():
    """Créer ou reconstruire le training_data"""
    parse_lexique('../lexiques.txt')
    build_training_data()
    charger_training_data()

def entrainer_modele(modele, nom, df_train):
    """Sous-menu pour entraîner un modèle donné"""
    os.makedirs("models/opt_results", exist_ok=True)
    print(f"\n--- Entraînement {nom} ---")
    print("1) Hyperparamètres basiques")
    print("2) GridSearchCV (optimisation)")

    choix = input("Choisissez une option (1-2) : ").strip()
    if choix == "1":
        modele.train(df_train)
    elif choix == "2":
        path = f"models/opt_results/{nom.lower().replace(' ', '_')}_optimise_model.pkl"
        modele.train(df_train, optimise=1, model_path=path)
    else:
        print("Option invalide, retour au menu.")

def evaluer_modele(modele, nom, df_val):
    """Sous-menu pour évaluer un modèle donné"""
    print(f"\n--- Évaluation {nom} ---")
    print("1) Modèle basique")
    print("2) Modèle GridSearchCV (optimisé)")

    choix = input("Choisissez une option (1-2) : ").strip()
    if choix == "1":
        modele.evaluate(df_val)
    elif choix == "2":
        path = f"models/opt_results/{nom.lower().replace(' ', '_')}_optimise_model.pkl"
        modele.evaluate(df_val, model_path=path)
    else:
        print("Option invalide, retour au menu.")

def menu():
    global train_df, val_df
    while True:
        print("\n--- MENU PRINCIPAL ---")
        print("1) Construire / charger training_data")
        print("2) Entraîner un ou plusieurs modèles")
        print("3) Évaluer un ou plusieurs modèles")
        print("4) Comparer les modèles")
        print("5) Quitter")

        choix = input("Choisissez une option (1-4) : ").strip()

        if choix == "1":
            if os.path.exists("../training_data.csv"):
                print("Le fichier training_data.csv existe déjà")
                print("1) Recréer le fichier et charger le DataFrame")
                print("2) Garder le fichier existant et charger le DataFrame")
                print("3) Revenir au menu principal")
                choix_file = input("Choisissez une option (1-3) : ").strip()
                if choix_file == "1":
                    construire_training_data()
                elif choix_file == "2":
                    charger_training_data()
                elif choix_file == "3":
                    continue
                else:
                    print("Option invalide, retour au menu.")
            else:
                construire_training_data()

        elif choix == "2":
            if train_df is None:
                print("ATTENTION -> Data non chargées. Veuillez d'abord construire ou charger training_data.")
                continue

            print("\nQuel modèle souhaitez-vous entraîner ?")
            model_dict = {
                "1": (adaboost, "AdaBoost"),
                "2": (bagging, "Bagging"),
                "3": (arbre_de_decision, "Arbre de Décision"),
                "4": (gboost, "GBoost"),
                "5": (random_forest, "Random Forest"),
                "6": None
            }
            for k, v in model_dict.items():
                if v: print(f"{k}) {v[1]}")
                else: print(f"{k}) Revenir au menu principal")

            choix_model = input("Choisissez une option (1-6) : ").strip()
            if choix_model in model_dict and model_dict[choix_model]:
                modele, nom = model_dict[choix_model]
                entrainer_modele(modele, nom, train_df)
            elif choix_model == "6":
                continue
            else:
                print("Option invalide, retour au menu.")

        elif choix == "3":
            if val_df is None:
                print("ATTENTION -> Data non chargées. Veuillez d'abord construire ou charger training_data.")
                continue

            print("\nQuel modèle souhaitez-vous évaluer ?")
            model_dict = {
                "1": (adaboost, "AdaBoost"),
                "2": (bagging, "Bagging"),
                "3": (arbre_de_decision, "Arbre de Décision"),
                "4": (gboost, "GBoost"),
                "5": (random_forest, "Random Forest"),
                "6": None
            }
            for k, v in model_dict.items():
                if v: print(f"{k}) {v[1]}")
                else: print(f"{k}) Revenir au menu principal")

            choix_model = input("Choisissez une option (1-6) : ").strip()
            if choix_model in model_dict and model_dict[choix_model]:
                modele, nom = model_dict[choix_model]
                evaluer_modele(modele, nom, val_df)
            elif choix_model == "6":
                continue
            else:
                print("Option invalide, retour au menu.")

        elif choix == "4":
            if val_df is None:
                print("ATTENTION -> Data non chargées. Veuillez d'abord construire ou charger training_data.")
                continue
            comparer_modeles(val_df)

        elif choix == "5":
            print("Au revoir !")
            break

        else:
            print("Option invalide, réessayez.")

if __name__ == "__main__":
    menu()
