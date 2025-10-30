from src.pipeline import run_preprocessing, run_model

if __name__ == "__main__":
    print("Lancement du pipeline de preprocessing...")
    run_preprocessing()
    print("")

    print("\nEntrainnement du modele...")
    model, acc, roc, y_proba = run_model()
    print("Entraînement terminé avec succès.\n")





