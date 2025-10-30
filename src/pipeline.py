import os
import pandas as pd
from . import config
from src.preprocessing import split_data, impute_det, impute_knn_hotdeck
from src.models import model_trainning

def run_preprocessing():
    """
    Script principal pour orchestrer l'ensemble du pipeline de preprocessing.
    1. Charge les données brutes.
    2. Divise en train/test.
    3. Exécute l'imputation déterministe.
    4. Exécute l'imputation stochastique k-NN.
    
    """
    # Vérification si les fichiers existent déjà
    if os.path.exists(config.train_path) and os.path.exists(config.test_path):
        print(f"Les fichiers prétraités existent déjà dans {config.processed_path}")
        print(" Étape de preprocessing ignorée.\n")
        train_set_imp = pd.read_parquet(config.train_path)
        test_set_imp = pd.read_parquet(config.test_path)
        return train_set_imp, test_set_imp
    else:
        print("Chargement des données brutes...")
        try:
            data_raw = pd.read_csv(config.data_path) 
        except FileNotFoundError:
            print(f"Erreur: Fichier de données non trouvé à l'emplacement: {config.data_path}")
            return

        print("Division Train/Test (stratifiée)...")
        train_set, test_set = split_data(
            data_raw,
            y=config.y,
            Test_size=config.Test_size,
            random_state=config.random_state
        )

        print("Etape 1: Imputation déterministe (Moyenne/Mode)...")
        train_set_det, test_set_det = impute_det(
            train_set, 
            test_set, 
            var=config.features 
        )

        print("Etape 2: Imputation k-NN (Hot-Deck)...")
        train_set_imp, test_set_imp = impute_knn_hotdeck(
            train_set_det, 
            test_set_det, 
            target_vars=config.var_to_imput, 
            numeric_features=config.aux_var_num,   
            categorical_features_ordinal=config.aux_var_ord,
            grade_order=config.grade,
            categorical_features_nominal=config.aux_var_nom,
            k_neighbors=config.k_neigh,
            random_state=config.random_state
        )

        print("Preprocessing terminé.")
        os.makedirs(config.processed_path, exist_ok=True)
        train_set_imp.to_parquet(config.train_path)
        test_set_imp.to_parquet(config.test_path)
        
        

def run_model():
    train_df = pd.read_parquet(config.train_path)
    test_df = pd.read_parquet(config.test_path)
    model, acc, roc, y_proba = model_trainning(train_df, test_df, save_dir=config.processed_path)
    print(f"Modèle sauvegardé dans {config.processed_path}/scoring_model.pkl")
    print("\nScript principal terminé avec succès.")
    return model, acc, roc, y_proba