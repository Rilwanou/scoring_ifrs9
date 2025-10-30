import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

def split_data(data_raw, y, Test_size, random_state):
    """
    Divise un DataFrame en jeux d'entraînement et de test de manière stratifiée.

    Args:
        data_raw (pd.DataFrame): Le DataFrame complet à diviser.
        y (str): Le nom de la colonne cible utilisée pour la stratification.
        Test_size (float): La proportion du jeu de données à inclure dans le test split.
        random_state (int): Le seed pour la reproductibilité.

    Returns:
        tuple: Un tuple contenant (train_set, test_set)
            - train_set (pd.DataFrame): Jeu d'entraînement.
            - test_set (pd.DataFrame): Jeu de test.
    """
    
    if y not in data_raw.columns:
        raise ValueError(f"La colonne de stratification '{y}' n'existe pas dans le DataFrame.")
        
    stratify_column = data_raw[y]
    
    train_set, test_set = train_test_split(
        data_raw,
        test_size=Test_size,
        random_state=random_state,
        stratify=stratify_column
    )
    
    return train_set, test_set

def impute_det(train_set, test_set, var):
    """
    Impute les variables déterministes (faible % de NaN)
    
    1. Apprend les valeurs (moyenne/mode) sur le train_set.
    2. Applique (transform) ces valeurs au train_set et au test_set.
    """
    train_set_imp = train_set.copy()
    test_set_imp = test_set.copy()

    per = round((train_set[var].isna().sum() / train_set.shape[0]) * 100, 2)
    na_per = {col: per[col] for col in var}
    col_fill = [p for p, v in na_per.items() if v <= 10 and v > 0]
    if 'annual_inc' not in col_fill and 'annual_inc' in train_set.columns:
        col_fill.append('annual_inc')

    num_cols = [n for n in col_fill if pd.api.types.is_numeric_dtype(train_set[n])]
    cat_cols = [ch for ch in col_fill if not pd.api.types.is_numeric_dtype(train_set[ch])]
    #  Imputation Numérique (Moyenne)
    if num_cols:
        imp_mean = SimpleImputer(strategy='mean')
        

        imp_mean.fit(train_set_imp[num_cols])
        
    
        train_set_imp[num_cols] = imp_mean.transform(train_set_imp[num_cols])
        test_set_imp[num_cols] = imp_mean.transform(test_set_imp[num_cols])

    #  Imputation Catégorielle (Mode)
    if cat_cols:
        imputer_mode = SimpleImputer(strategy='most_frequent')
        
     
        imputer_mode.fit(train_set_imp[cat_cols])
        
     
        train_set_imp[cat_cols] = imputer_mode.transform(train_set_imp[cat_cols])
        test_set_imp[cat_cols] = imputer_mode.transform(test_set_imp[cat_cols])

    return train_set_imp, test_set_imp
    
def impute_knn_hotdeck(train_set, test_set, target_vars, 
                       numeric_features, categorical_features_ordinal, 
                       grade_order, categorical_features_nominal, 
                       k_neighbors=10, random_state=44):
    """
    Impute les valeurs manquantes sur le train_set et test_set en utilisant 
    une méthode k-NN stochastique (hot-deck).

    La logique de non-fuite de données est respectée :
    1. Le preprocessor est fitté UNIQUEMENT sur 'train_set'.
    2. Les "donneurs" sont identifiés UNIQUEMENT dans 'train_set'.
    3. Le modèle k-NN est fitté UNIQUEMENT sur les donneurs du 'train_set'.
    4. Les receveurs de 'train_set' et 'test_set' sont imputés en utilisant 
       les donneurs du 'train_set'.
    5. Seules les valeurs NaN sont remplacées.

    Args:
        train_set (pd.DataFrame): DataFrame d'entraînement brut (avec NaN).
        test_set (pd.DataFrame): DataFrame de test brut (avec NaN).
        target_vars (list): Liste des noms de colonnes à imputer (ex: ['open_acc_6m', ...]).
        numeric_features (list): Liste des colonnes numériques pour le k-NN.
        categorical_features_ordinal (list): Liste des colonnes ordinales (ex: ['grade']).
        grade_order (list): Ordre des catégories pour la variable 'grade'.
        categorical_features_nominal (list): Liste des colonnes nominales (ex: ['home_ownership']).
        k_neighbors (int, optional): Nombre de voisins à considérer. Défaut à 10.
        random_state (int, optional): Seed pour la reproductibilité du choix aléatoire. Défaut à 44.

    Returns:
        tuple: Un tuple contenant (train_set_imp, test_set_imp)
            - train_set_imp (pd.DataFrame): DataFrame d'entraînement imputé.
            - test_set_imp (pd.DataFrame): DataFrame de test imputé.
    """


    np.random.seed(random_state)
    train_set_imp = train_set.copy()
    test_set_imp = test_set.copy()
    
    # Définition des variables auxiliaires et du preprocessor ---
    aux_vars = numeric_features + categorical_features_ordinal + categorical_features_nominal

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat_nom', OneHotEncoder(handle_unknown='ignore'), categorical_features_nominal),
            ('cat_ord', OrdinalEncoder(categories=[grade_order]), categorical_features_ordinal)
        ],
        remainder='passthrough'
    )


    preprocessor.fit(train_set[aux_vars])
    
    X_train_processed = preprocessor.transform(train_set[aux_vars])
    X_test_processed = preprocessor.transform(test_set[aux_vars])

    # Identification des Donneurs (TRAIN) et Receveurs (TRAIN + TEST) ---
    is_donor_train = train_set[target_vars].notna().all(axis=1)
    is_recipient_train = ~is_donor_train
    
    # Un receveur de test est une ligne avec au moins un NaN dans les cibles
    is_recipient_test = test_set[target_vars].isna().any(axis=1)



    df_donor = train_set[is_donor_train]
    df_donor_processed = X_train_processed[is_donor_train]

    X_train_recip = X_train_processed[is_recipient_train]
    train_recip_idx = train_set.index[is_recipient_train] # Index originaux

    X_test_recip = X_test_processed[is_recipient_test]
    test_recip_idx = test_set.index[is_recipient_test] # Index originaux

    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', n_jobs=-1)
    nn_model.fit(df_donor_processed)

    if not X_train_recip.shape[0] == 0:
        dist_train, idx_train = nn_model.kneighbors(X_train_recip)

        n_recip_train = len(idx_train)
        k_neighbors_train = idx_train.shape[1]
        
        random_donor_idx_train = np.random.randint(0, k_neighbors_train, size=n_recip_train)
        sorted_idx_train = np.arange(n_recip_train)

        donor_idx_train = idx_train[sorted_idx_train, random_donor_idx_train]
        bloc_train = df_donor.iloc[donor_idx_train]
        
        # Aligner l'index du bloc sur celui des receveurs pour le fillna
        bloc_train.index = train_recip_idx 

        for p in target_vars:
            train_set_imp[p] = train_set_imp[p].fillna(bloc_train[p])

    # Imputation du TEST SET (en utilisant les donneurs du TRAIN SET) ---
    if not X_test_recip.shape[0] == 0:
        dist_test, idx_test = nn_model.kneighbors(X_test_recip)

        n_recip_test = len(idx_test)
        k_neighbors_test = idx_test.shape[1]

        random_donor_idx_test = np.random.randint(0, k_neighbors_test, size=n_recip_test)
        sorted_idx_test = np.arange(n_recip_test)

        donor_idx_test = idx_test[sorted_idx_test, random_donor_idx_test]
        
        # Le bloc de donneurs vient toujours de df_donor (TRAIN SET)
        bloc_test = df_donor.iloc[donor_idx_test]
        
        # Aligner l'index du bloc sur celui des receveurs
        bloc_test.index = test_recip_idx

        for p in target_vars:
            test_set_imp[p] = test_set_imp[p].fillna(bloc_test[p])

    return train_set_imp, test_set_imp