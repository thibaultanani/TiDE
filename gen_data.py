import random
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def write(df, name, save=True):
    if save:
        # Enregistrer les données en tant que fichiers Excel
        filename = 'datasets/' + str(name)
        df.to_excel(filename + '.xlsx', index=False)
        print('Données de simulations ' + filename + ' avec ' + str(df.shape[1] - 1) +
              ' variables générées et sauvegardées avec succès.')
    correlations = df.corrwith(df["target"], method='spearman').drop("target")
    # print("correlations", correlations.to_string())
    return df


def write_(data, filename, n_features):
    filename = 'datasets/' + str(filename)
    data.to_excel(filename + '_train.xlsx', index=False)
    print('Données de simulations ' + filename + ' avec ' + str(n_features) +
          ' variables générées et sauvegardées avec succès.')


def create_dataset(n_samples=3000, n_features=100, random_seed=None, relevant=10, noise_level=0.0):
    # Fixer la graine aléatoire pour la reproductibilité
    if random_seed is not None:
        np.random.seed(random_seed)
    # Générer les variables aléatoires
    X = np.random.randn(n_samples, n_features)
    # Normaliser les variables
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Créer la variable cible comme une combinaison linéaire des 10 premières variables
    weights = np.random.randn(relevant)
    y = X[:, :relevant].dot(weights)
    # Afficher les poids et la formule de y
    print("Poids (weights) :", weights)
    for i, weight in enumerate(weights):
        print(f"w{i+1} * f{i+1}", end=" + " if i < len(weights) - 1 else "\n")
    # Ajouter du bruit si noise_level > 0
    if noise_level > 0:
        noise = noise_level * np.random.randn(n_samples)
        y += noise
    # Binariser la variable cible
    y = (y > np.median(y))
    # Créer un DataFrame
    df = pd.DataFrame(X, columns=[f'f{i + 1}' for i in range(n_features)])
    df['target'] = y
    return df


def add_class_noise(df, target_feature, n):
    """
    Ajoute du bruit à la variable cible en inversant un certain pourcentage de ses valeurs.

    :param df: DataFrame contenant le jeu de données.
    :param target_feature: Nom de la variable cible.
    :param n: Pourcentage de valeurs à inverser (float entre 0 et 1).
    :return: DataFrame avec la variable cible bruitée.
    """
    df_noisy = df.copy()
    n_samples = len(df_noisy)
    n_noisy_samples = int(n * n_samples)
    np.random.seed(42)
    noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
    df_noisy[target_feature] = df_noisy[target_feature].astype(int)
    df_noisy.loc[noisy_indices, target_feature] = 1 - df_noisy.loc[noisy_indices, target_feature]
    df_noisy[target_feature] = df_noisy[target_feature].astype(bool)
    return df_noisy


def add_feature_noise(df, target_feature, n):
    """
    Ajoute n variables aléatoires au jeu de données et déplace la variable cible à la fin.

    :param df: DataFrame contenant le jeu de données.
    :param target_feature: Nom de la variable cible.
    :param n: Nombre de variables aléatoires à ajouter.
    :return: DataFrame avec n variables aléatoires ajoutées et la variable cible à la fin.
    """
    df_extended = df.copy()
    n_samples = len(df_extended)
    np.random.seed(42)
    random_vars = np.random.randn(n_samples, n)
    random_df = pd.DataFrame(random_vars, columns=[f'random_var_{i + 1}' for i in range(n)])
    df_extended = pd.concat([df_extended, random_df], axis=1)
    target_data = df_extended.pop(target_feature)
    df_extended[target_feature] = target_data
    return df_extended


def add_redundancy(df, target_feature, n):
    """
    Duplique les 10 premières variables du DataFrame.

    :param df: DataFrame contenant le jeu de données.
    :param target_feature: Nom de la variable cible.
    :param n: Nombre de variables aléatoires à dupliquer.
    :return: DataFrame avec les n premières variables dupliquées.
    """
    df_extended = df.copy()
    first_n_columns = df_extended.columns[:n]
    for col in first_n_columns:
        df_extended[f'{col}_duplicate'] = df_extended[col]
    target_data = df_extended.pop(target_feature)
    df_extended[target_feature] = target_data
    return df_extended


def add_imbalance(df, target_feature, ratio):
    """
    Déséquilibre le jeu de données pour un certain pourcentage de la classe majoritaire.

    :param df: DataFrame contenant le jeu de données.
    :param target_feature: Nom de la variable cible.
    :param ratio: Ratio de déséquilibre (0 < imbalance_ratio < 1).
    :return: DataFrame déséquilibré.
    """
    if ratio < 0 or ratio > 0.5:
        raise ValueError("Le ratio doit être entre 0 et 0.5")
    class_0 = df[df[target_feature] == 0]
    class_1 = df[df[target_feature] == 1]
    n_class_0 = len(class_0)
    n_class_1 = len(class_1)
    n_class_0_new = int(n_class_1 * ratio / (1 - ratio))
    class_0_reduced = class_0.sample(n=n_class_0_new, random_state=42)
    balanced_df = pd.concat([class_0_reduced, class_1])
    return balanced_df


if __name__ == '__main__':
    # Baseline Dataset
    data = create_dataset(n_samples=5000, n_features=100, random_seed=42, relevant=10, noise_level=1)
    write(df=data, name="baseline")
    print(data)

    class_noise = add_class_noise(df=data, target_feature='target', n=0.2)
    write(df=class_noise, name="class_noise")

    feature_noise = add_feature_noise(df=data, target_feature='target', n=400)
    write(df=feature_noise, name="feature_noise")

    redundancy = add_redundancy(df=data, target_feature='target', n=10)
    write(df=redundancy, name="redundancy")

    imbalanced = add_imbalance(df=data, target_feature='target', ratio=0.35)
    write(df=imbalanced, name="imbalanced")

