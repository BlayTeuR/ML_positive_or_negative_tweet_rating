# Ce fichier effectue un prétraitement sur les données avant que celles-ci soient passées aux modèles.
# Le but est de faciliter le traitement des données et de réduire le bruit

import pandas as pd
import re

# Cette fonction normalise les mots
# elle remplace les abréviations ou mots familiers par leur forme standard
# réduit les répétitions exagérées de lettre
# le paramètre 'df' représente le dataframe en mémoire
def normalisation_df(df):
    df = df.copy()
    df["SentimentText"] = df["SentimentText"].astype(str)
    
    lexique_abbrev = {
        "lol": "laughing",
        "omg": "oh my god",
        "btw": "by the way",
        "idk": "i don't know",
        "lmao": "laughing",
        "rofl": "rolling on the floor laughing",
        "smh": "shaking my head",
        "tbh": "to be honest",
        "afaik": "as far as i know",
        "brb": "be right back",
        "fyi": "for your information",
        "imo": "in my opinion",
        "jk": "just kidding",
        "np": "no problem"
    }

    def normaliser_texte(texte):
        texte = texte.lower()
        # Remplacer les abréviations par leur forme standard
        for abbr, full in lexique_abbrev.items():
            texte = re.sub(rf"\b{abbr}\b", full, texte)
        # Réduire les lettres répétées plus de 2 fois consécutives
        texte = re.sub(r"(.)\1{2,}", r"\1\1", texte)
        # Supprimer les espaces multiples restants
        texte = re.sub(r"\s+", " ", texte).strip()
        return texte

    df["SentimentText"] = df["SentimentText"].apply(normaliser_texte)
    return df

# Cette fonction effectue un nettoyage de base :
# - suppression des URL
# - retrait du symbole '#'
# - suppression des mentions 'mot commençant pas @'
# le paramètre 'df' représente le dataframe en mémoire
def nettoyage_df(df):
    df = df.copy()
    df["SentimentText"] = df["SentimentText"].astype(str)
    
    def nettoyer_texte(texte):
        if pd.isna(texte):
            return ""
        # Supprimer les URL (http, https, www)
        texte = re.sub(r"http\S+|www\S+|https\S+", "", texte, flags=re.MULTILINE)
        # Supprimer les mentions @user
        texte = re.sub(r"@\w+", "", texte)
        # Normaliser les hashtags #mot -> mot
        texte = re.sub(r"#(\w+)", r"\1", texte)
        # Supprimer les espaces multiples restants
        texte = re.sub(r"\s+", " ", texte).strip()
        return texte

    df["SentimentText"] = df["SentimentText"].apply(nettoyer_texte)
    return df