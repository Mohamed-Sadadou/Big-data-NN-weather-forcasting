# Big-data-NN-weather-forecasting

## Description

Ce projet implémente un système de prédiction météorologique utilisant des réseaux de neurones pour l'analyse de big data. Il s'agit d'un projet académique de Master 1 VMI (Vision Machine Intelligence) axé sur la prédiction du rayonnement solaire à courte longueur d'onde (shortwave radiation) à partir de données météorologiques historiques.

## Objectif

Développer et comparer différentes architectures de réseaux de neurones pour la prédiction météorologique, en particulier pour estimer les valeurs de rayonnement solaire basées sur des paramètres météorologiques historiques.

## Architecture du Projet

### Classes Principales

#### `NeuralNet` (Classe Abstraite)
Classe de base abstraite définissant l'interface commune pour tous les modèles de réseaux de neurones.

**Paramètres par défaut :**
- `MAX_EPOCHS`: 1000
- `BATCH_SIZE`: 128
- `LOSS`: 'mean_absolute_error'
- `METRICS`: ['mean_squared_error']
- `LEARNING_RATE`: 0.1
- `ACTIVATION`: 'sigmoid'
- `NNEURONS`: [8]

**Méthodes principales :**
- `build_model()`: Méthode abstraite à implémenter
- `train_model_with_valid()`: Entraînement avec validation
- `train_model()`: Entraînement simple
- `predict()`: Prédiction sur nouvelles données

#### `MLP` (Multi-Layer Perceptron)
Implémentation d'un perceptron multicouche avec architecture dense.

**Caractéristiques :**
- Couche d'aplatissement (Flatten)
- Couches denses configurables
- Activation linéaire en sortie

#### `LSTMc` (LSTM Custom)
Réseau de neurones récurrent LSTM pour la prédiction de séries temporelles.

**Caractéristiques :**
- Support multi-couches LSTM
- Gestion des séquences temporelles
- Architecture flexible selon le nombre de neurones

#### `ArbitraryNN`
Architecture personnalisée combinant LSTM, Dropout et BatchNormalization.

**Architecture :**
- 2 couches LSTM (8 neurones, activation tanh)
- Dropout (0.1) et BatchNormalization
- Couche dense finale

## Dataset

Le projet utilise un dataset de données météorologiques historiques (`Historical weahe.csv`) contenant :
- 18 variables météorologiques en entrée
- Variable cible : rayonnement solaire à courte longueur d'onde
- Dataset de test : 137,261 échantillons

## Évaluation des Performances

Le système évalue les modèles avec les métriques suivantes :

| Métrique | Description |
|----------|-------------|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| nMAE | Normalized MAE |
| nRMSE | Normalized RMSE |

## Visualisation

Le projet inclut des graphiques de comparaison entre :
- Valeurs mesurées (rouge)
- Valeurs prédites (bleu)

## Technologies Utilisées

- **TensorFlow/Keras** : Framework de deep learning
- **NumPy** : Calculs numériques
- **Pandas** : Manipulation de données
- **Scikit-learn** : Métriques d'évaluation et préprocessing
- **Matplotlib** : Visualisation
- **Google Colab** : Environnement de développement

## Structure des Fichiers

```
Big-data-NN-weather-forecasting/
├── BigDataProject.ipynb    # Notebook principal du projet
├── README.md              # Documentation du projet
├── report.pdf            # Rapport détaillé du projet
└── Historical weahe.csv  # Dataset météorologique
```

## Installation et Utilisation

### Prérequis
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### Utilisation
1. Charger le dataset météorologique
2. Instancier le modèle souhaité (MLP, LSTMc, ou ArbitraryNN)
3. Construire le modèle avec `build_model(input_shape)`
4. Entraîner avec `train_model()` ou `train_model_with_valid()`
5. Prédire avec `predict()`
6. Évaluer les performances avec les métriques fournies

### Exemple d'usage
```python
# Créer et configurer le modèle
mlp = MLP(nneurons=[32, 16], activation='relu')
mlp.build_model(input_shape=(18,))

# Entraîner le modèle
mlp.train_model(X_train, y_train, epochs=100)

# Prédire
predictions = mlp.predict(X_test)
```

## Résultats

Le projet démontre l'efficacité des réseaux de neurones pour la prédiction météorologique, avec une évaluation complète des performances sur un large dataset de test.

## Auteur

**Mohamed Sadadou**  
Projet M1 VMI - Big Data et Intelligence Artificielle

## Licence

Ce projet est développé dans un cadre académique pour l'apprentissage des techniques de big data et de machine learning appliquées à la météorologie.
