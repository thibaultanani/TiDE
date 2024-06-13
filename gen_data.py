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
    print("correlations", correlations.to_string())
    return df


def write_(data, filename, n_features):
    filename = 'datasets/' + str(filename)
    data.to_excel(filename + '_train.xlsx', index=False)
    print('Données de simulations ' + filename + ' avec ' + str(n_features) +
          ' variables générées et sauvegardées avec succès.')


def create_dataset(n_samples=3000, n_features=100, random_seed=None, noise_level=0.0):
    # Fixer la graine aléatoire pour la reproductibilité
    if random_seed is not None:
        np.random.seed(random_seed)
    # Générer les variables aléatoires
    X = np.random.randn(n_samples, n_features)
    # Normaliser les variables
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Créer la variable cible comme une combinaison linéaire des 10 premières variables
    weights = np.random.randn(10)
    y = X[:, :10].dot(weights)
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


if __name__ == '__main__':
    # Baseline Dataset
    data = create_dataset(n_samples=3000, n_features=100, random_seed=42, noise_level=1)
    write(df=data, name="baseline")
    print(data)