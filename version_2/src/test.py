import joblib
import pandas as pd 
import data_preparation

def build_test_data(csv="../test.csv"):
    df = pd.read_csv(csv, encoding="ISO-8859-1")

    colonnes = [
        "id_tweet", 
        "score_mots_pos_neg", 
        "score_emojis_pos_neg", 
        "nb_exclamations_intensifieurs",
        "nb_negations",
        "nb_moycaracteres",
        "nb_mots",
    ]

    lignes = []
    id_tweet = 1

    for index, row in df.iterrows():
        try:
            tweet = row["SentimentText"]
            score_mot = data_preparation.score_mots_pos_neg(tweet)
            score_emojis = data_preparation.score_emojis_pos_neg(tweet)
            nb_excla_intensifieur = data_preparation.nb_exclamations_intensifieurs(tweet)
            nb_nega = data_preparation.nb_negations(tweet)
            nb_moy_cara = data_preparation.nb_moycaracteres(tweet)
            nb_mot = data_preparation.nb_mots(tweet)

            lignes.append({
                "id_tweet": id_tweet, 
                "score_mots_pos_neg": score_mot, 
                "score_emojis_pos_neg": score_emojis, 
                "nb_exclamations_intensifieurs": nb_excla_intensifieur,
                "nb_negations": nb_nega,
                "nb_moycaracteres": nb_moy_cara,
                "nb_mots": nb_mot,
            })

            id_tweet += 1
            
        except Exception as e:
            print(f"Erreur à la ligne {index}: {e}")
            continue

    print("✅ Nombre total de lignes traitées :", len(lignes))
    df_features = pd.DataFrame(lignes, columns=colonnes)
    df_features.to_csv("../test_data.csv", index=False, encoding="ISO-8859-1")
    return df_features

def build_prediction():

    # Chargement du modèle choisis
    model_path = "models/opt_results/adaboost_optimise_model.pkl"
    model = joblib.load(model_path)
    print("Modèle chargé :", model_path)

    # Chargement des features de test
    test_features = pd.read_csv("../test_data.csv", encoding="ISO-8859-1")
    print("Données de test chargées :", test_features.shape)

    # Chargement du texte du tweet pour l'associer aux prédiction
    test_original = pd.read_csv("../test.csv", encoding="ISO-8859-1")

    X_test = test_features.drop(columns=["id_tweet"])
    y_pred = model.predict(X_test)

    # Construction du DataFrame
    output = pd.DataFrame({
        "ItemID": test_features["id_tweet"],
        "SentimentText": test_original["SentimentText"],
        "PredictedSentiment": y_pred
    })

    output.to_csv("../test_predictions.csv", index=False, encoding="ISO-8859-1")
    print("Fichier de prédictions sauvegardé dans: src/test_predictions.csv")

build_test_data()
build_prediction()

