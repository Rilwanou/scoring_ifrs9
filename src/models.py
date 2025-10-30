import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from src.config import num_features, cat_features, cat_order
from . import config

def model_trainning(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir=config.processed_path):
# --- Mapping des labels ---
    mapping = {'Fully Paid': 0, 'Charged Off': 1}
    train_df["loan_status"] = train_df["loan_status"].map(mapping)
    test_df["loan_status"] = test_df["loan_status"].map(mapping)

    # --- Split train/test ---
    X, y = train_df.drop("loan_status", axis=1), train_df["loan_status"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Pipeline ---
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    cat_transformer = Pipeline(steps=[("ord_enc", OrdinalEncoder(categories=cat_order))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])

    # --- Entraînement ---
    model.fit(X_train, y_train)

    # --- Prédiction ---
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    # --- Scores ---
    acc = accuracy_score(y_valid, y_pred)
    roc = roc_auc_score(y_valid, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")

    # --- Sauvegarde ---
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "scoring_model.pkl"))

    return model, acc, roc, y_proba
