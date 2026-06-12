# Ce fichier prépare les données pour l'analyse de sentiment des tweets
# Il fournit des fonctions pour calculer différentes caractéristiques textuelles et émotionnelles

from pre_traitement_donnees import *
import string
import pandas as pd

# La fonction parse_lexique lit le fichier lexiques.txt et organise les mots et emojis en ensembles (sets) par catégorie.
# Chaque catégorie correspond à un type de signal émotionnel ou linguistique :
# - mots_positifs / mots_negatifs : lexique de mots indiquant un sentiment positif ou négatif
# - emojis_positifs / emojis_negatifs : emojis indiquant un sentiment positif ou négatif
# - intensifieurs : mots amplifiant l'émotion ("very", "so", etc.)
# - negations : mots niant ou inversant le sens ("not", "no")
# - mots_ambivalents : mots pouvant être positifs ou négatifs selon le contexte
# Les sets permettent un accès rapide pour vérifier si un mot ou emoji appartient à une catégorie.
def parse_lexique(path):
    lexiques = {
        "mots_positifs": set(),
        "mots_negatifs": set(),
        "emojis_positifs": set(),
        "emojis_negatifs": set(),
        "intensifieurs": set(),
        "negations": set(),
        "mots_ambivalents": set()
    }

    i = -1  # compteur pour savoir dans quelle catégorie nous sommes lors de la lecture

    with open(path, 'r', encoding="utf-8") as f:
        lignes = f.readlines()

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue  # ignorer les lignes vides
        if ligne.startswith("###"):
            i += 1  # nouvelle catégorie
            continue
        else:
            # Ajouter le mot ou emoji dans le set correspondant à la catégorie actuelle
            match i:
                case 0:
                    lexiques["mots_positifs"].add(ligne)
                case 1:
                    lexiques["mots_negatifs"].add(ligne)
                case 2:
                    lexiques["emojis_positifs"].add(ligne)
                case 3:
                    lexiques["emojis_negatifs"].add(ligne)
                case 4:
                    lexiques["intensifieurs"].add(ligne)
                case 5:
                    lexiques["negations"].add(ligne)
                case 6:
                    lexiques["mots_ambivalents"].add(ligne)

    return lexiques

lexiques = parse_lexique('../lexiques.txt')

# Calcul du score basé sur les mots positifs et négatifs
# Pour chaque mot dans le tweet :
# - +1 si c'est un mot positif
# - -1 si c'est un mot négatif
# Ce score donne une première approximation du sentiment global du tweet.
def score_mots_pos_neg(tweet):
    score = 0
    mots = tweet.split(' ')
    for mot in mots:
        mot_clean = mot.lower().strip(string.punctuation)
        if mot_clean in lexiques["mots_positifs"]:
            score += 1
        elif mot_clean in lexiques["mots_negatifs"]:
            score -= 1
    return score

# Calcul du score basé sur les emojis positifs et négatifs
# Les emojis sont souvent des indicateurs forts d'émotion donc leur présence influence directement le score
def score_emojis_pos_neg(tweet):
    score = 0
    mots = tweet.split(' ')
    for mot in mots:
        mot_clean = mot.lower().strip(string.punctuation)
        if mot_clean in lexiques["emojis_positifs"]:
            score += 1
        elif mot_clean in lexiques["emojis_negatifs"]:
            score -= 1
    return score

# Compte les exclamations et les intensifieurs pour mesurer l'intensité émotionnelle
# Les points d'exclamation et les amplificateurs augmentent la force du sentiment exprimé
def nb_exclamations_intensifieurs(tweet):
    score = tweet.count('!')
    mots = tweet.split(' ')
    for mot in mots:
        mot_clean = mot.lower().strip(string.punctuation)
        if mot_clean in lexiques["intensifieurs"]:
            score += 1
    return score

# Compte le nombre de mots de négation dans le tweet
# Les mots de négation peuvent inverser le sens d'une phrase et sont donc essentiels pour détecter le sentiment réel
def nb_negations(tweet):
    score = 0
    mots = tweet.split(' ')
    for mot in mots:
        mot_clean = mot.lower().strip(string.punctuation)
        if mot_clean in lexiques["negations"]:
            score += 1
    return score

# Compte le nombre total de mots dans le tweet
# Cette métrique est utile pour normaliser d'autres scores ou détecter des tweets très courts ou longs
def nb_mots(tweet):
    return len(tweet.split())

# Calcule la longueur moyenne des mots dans le tweet
# Les mots courts sont souvent associés à un ton informel ou positif tandis que les mots plus longs peuvent indiquer un ton sérieux ou négatif
# Les mots sont nettoyés de la ponctuation avant le calcul.
def nb_moycaracteres(tweet):
    somme = 0
    nbMots = 0
    mots = tweet.split(' ')
    for mot in mots:
        mot_clean = mot.lower().strip(string.punctuation)
        if mot_clean:
            somme += len(mot_clean)
            nbMots += 1
    return somme / nbMots if nbMots > 0 else 0

# Construit la table de données (training_data.csv) à partir des tweets (train.csv)
def build_training_data(csv="../train.csv"):
    df = pd.read_csv(csv, encoding="ISO-8859-1")
    
    # Prétraitement en mémoire
    df = nettoyage_df(df)
    df = normalisation_df(df)

    colonnes = [
        "id_tweet", 
        "score_mots_pos_neg", 
        "score_emojis_pos_neg", 
        "nb_exclamations_intensifieurs",
        "nb_negations",
        "nb_moycaracteres",
        "nb_mots",
        "target"
    ]    

    lignes = []
    id_tweet = 1

    for index, row in df.iterrows():
        try:
            tweet = row["SentimentText"]
            score_mot = score_mots_pos_neg(tweet)
            score_emojis = score_emojis_pos_neg(tweet)
            nb_excla_intensifieur = nb_exclamations_intensifieurs(tweet)
            nb_nega = nb_negations(tweet)
            nb_moy_cara = nb_moycaracteres(tweet)
            nb_mot = nb_mots(tweet)
            target = row["Sentiment"]

            lignes.append({
                "id_tweet": id_tweet, 
                "score_mots_pos_neg": score_mot, 
                "score_emojis_pos_neg": score_emojis, 
                "nb_exclamations_intensifieurs": nb_excla_intensifieur,
                "nb_negations": nb_nega,
                "nb_moycaracteres": nb_moy_cara,
                "nb_mots": nb_mot,
                "target": target
            })

            id_tweet += 1

        except Exception as e:
            print(f"Erreur à la ligne {index}: {e}")
            continue

    print("✅ Nombre total de lignes traitées :", len(lignes))

    df_features = pd.DataFrame(lignes, columns=colonnes)
    
    # retourner le DataFrame et sauvegarder
    df_features.to_csv("../training_data.csv", index=False, encoding="ISO-8859-1")
    return df_features