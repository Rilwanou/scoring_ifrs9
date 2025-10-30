# ÉTAPE 1: Le "Builder" - Installe les dépendances
FROM python:3.13-slim as builder

WORKDIR /app

# Installe Poetry
RUN pip install poetry

# Copie les fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# --- CORRECTION ICI ---
# On force Poetry à créer le dossier .venv DANS /app
RUN poetry config virtualenvs.in-project true

# Installe les dépendances (SAUF le groupe 'dev')
# Cela va maintenant créer /app/.venv
RUN poetry install --no-root --without dev


# ÉTAPE 2: L'image "Finale" - Propre et légère
# (Renommé 'stage-1' en 'final' pour plus de clarté)
FROM python:3.13-slim as final

WORKDIR /app

# Copie l'environnement virtuel installé depuis l'étape "builder"
# Cette commande va maintenant fonctionner
COPY --from=builder /app/.venv /app/.venv

# Active l'environnement virtuel pour toutes les commandes suivantes
ENV PATH="/app/.venv/bin:$PATH"

# --- MODIFICATIONS POUR STREAMLIT ---

# 1. Copier le script du dashboard
COPY dashboard.py .

# 2. Copier le modèle (grâce à l'exception du .dockerignore)
COPY data/processed/scoring_model.pkl data/processed/scoring_model.pkl

# 3. Exposer le port standard de Streamlit
EXPOSE 8501

# 4. Commande de démarrage : Lancer Streamlit
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]