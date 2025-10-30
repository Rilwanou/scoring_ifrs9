# dashboard.py

import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Simulateur ECL (IFRS 9)",
    page_icon="📊",
    layout="wide"  # "wide" donne plus d'espace
)
st.title("📊 Simulateur de Risque de Crédit (IFRS 9)")

# --- 2. CHARGEMENT DU MODÈLE PD ---
@st.cache_resource
def load_model():
    """Charge le pipeline de scoring (modèle PD) sauvegardé."""
    model_path = os.path.join("data", "processed", "scoring_model.pkl") 
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Erreur: Fichier modèle non trouvé à: {model_path}")
        return None

model_pd = load_model()

# --- 3. DÉFINITION DES HYPOTHÈSES (LGD) ---
LGD = 0.45 
st.info(f"Hypothèse de LGD (Loss Given Default) fixée à : **{LGD*100:.0f}%** (standard pour prêt non garanti)")


# --- 4. INTERFACE UTILISATEUR (INPUTS) ---

# On divise l'interface en 2 colonnes : Inputs | Résultats
col_inputs, col_results = st.columns(2)

with col_inputs:
    st.header("Paramètres du Client et du Prêt")

    # A. Input EAD (le plus important pour ECL)
    st.subheader("Exposition (EAD)")
    # C'est votre 'out_prncp'
    ead_input = st.number_input(
        "Capital restant dû (€)", 
        0.0, 1000000.0, 25000.0, step=100.0,
        help="Correspond à 'out_prncp'. C'est l'Exposition en cas de Défaut (EAD)."
    )
    
    st.divider()

    # B. Inputs "Top Features" (pour PD)
    st.subheader("Caractéristiques Principales (PD)")
    
    # On utilise des colonnes pour mieux agencer
    c1, c2 = st.columns(2)
    with c1:
        annual_inc = st.number_input("Revenu annuel (€)", 1000, 5000000, 60000, step=1000)
        dti = st.number_input("Taux d'endettement (%)", 0.0, 100.0, 15.0, step=0.1)
        term = st.selectbox("Durée du prêt (mois)", [' 36 months', ' 60 months'])
        # Si votre modèle attend des strings, utilisez : [' 36 months', ' 60 months']
        
    with c2:
        installment = st.number_input("Mensualité estimée (€)", 0.0, 5000.0, 450.0, step=10.0)
        grade = st.selectbox("Notation (Grade) du prêt", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        int_rate = st.number_input("Taux d'intérêt (%)", 0.0, 30.0, 12.5, step=0.1)


    # C. Inputs "Secondaires" (cachés par défaut)
    st.divider()
    with st.expander("Afficher les paramètres avancés (Emprunteur & Comportement)"):
        
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("Profil Emprunteur")
            emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years','6 years', '7 years', '8 years', '9 years', '10+ years']
            emp_length = st.selectbox("Ancienneté professionnelle", options=emp_length_options, index=5)
            home_ownership = st.selectbox(
                "Situation immobilière", 
                ['MORTGAGE', 'RENT', 'OWN', 'OTHER'],
                help="MORTGAGE=Propriétaire (avec hypothèque), RENT=Locataire, OWN=Propriétaire"
            )
            mort_acc = st.number_input("Nombre de prêts immobiliers", 0, 20, 1, step=1)
            
        with c4:
            st.markdown("Comportement de Crédit")
            tot_cur_bal = st.number_input("Solde total tous comptes (€)", 0, 1000000, 150000, step=1000)
            open_acc_6m = st.number_input("Nouveaux crédits (6 derniers mois)", 0, 10, 0, step=1)
            total_bal_il = st.number_input("Solde total (prêts à tempérament)", 0, 500000, 50000, step=100)
            inq_fi = st.number_input("Demandes de crédit (6 derniers mois)", 0, 10, 0, step=1)
            num_sats = st.number_input("Nombre de crédits (sans incident)", 0, 50, 8, step=1)


# --- 5. CALCUL ET AFFICHAGE (OUTPUTS) ---
with col_results:
    st.header("Résultats de la Simulation")

    if st.button("Calculer le Risque (ECL)", type="primary", use_container_width=True):
        if model_pd is None:
            st.error("Le modèle n'a pas pu être chargé. Calcul impossible.")
        else:
            # A. Créer le DataFrame pour la prédiction
            # Doit contenir TOUTES les 12 features avec les BONS NOMS DE COLONNES
            try:
                client_data_dict = {
                    # Features principales
                    'term': [term],
                    'int_rate': [int_rate],
                    'installment': [installment],
                    'grade': [grade],
                    'annual_inc': [annual_inc],
                    'dti': [dti],
                    
                    # Features secondaires (cachées)
                    'emp_length': [emp_length],
                    'home_ownership': [home_ownership],
                    'tot_cur_bal': [tot_cur_bal],
                    'open_acc_6m': [open_acc_6m],
                    'total_bal_il': [total_bal_il],
                    'inq_fi': [inq_fi],
                    'mort_acc': [mort_acc],
                    'num_sats': [num_sats]
                }
                
                client_data = pd.DataFrame(client_data_dict)

                # B. Calculer la PD
                # On suppose que la classe "1" (défaut) est la deuxième
                pd_value = model_pd.predict_proba(client_data)[0, 1]
                
                # C. Calculer l'ECL
                ecl_value = pd_value * LGD * ead_input
                
                # D. Afficher les résultats
                st.subheader("Ventilation du Risque")
                
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                r1_c1.metric(
                    label="Probabilité de Défaut (PD)", 
                    value=f"{pd_value:.2%}",
                    help="Probabilité que l'emprunteur fasse défaut."
                )
                r1_c2.metric(
                    label="Exposition (EAD)", 
                    value=f"€ {ead_input:,.0f}",
                    help="Montant exposé en cas de défaut."
                )
                r1_c3.metric(
                    label="Perte (LGD)", 
                    value=f"{LGD:.0%}",
                    help="Part de l'exposition qui sera perdue (hypothèse)."
                )
                
                st.divider()
                
                st.metric(
                    label="Provision ECL (Expected Credit Loss)",
                    value=f"€ {ecl_value:,.2f}",
                    help=f"Formule: ECL = PD * LGD * EAD\n({pd_value:.2%} * {LGD} * {ead_input:,.0f})"
                )
                
                # Bonus : Jauge de Risque
                st.progress(pd_value, text=f"Niveau de risque (PD): {pd_value:.2%}")


            except Exception as e:
                st.error(f"Erreur lors de la prédiction. Vérifiez vos features.")
                st.error(f"Détail : {e}")
                st.warning("Assurez-vous que le modèle a été entraîné avec "
                         "exactement ces 12 noms de colonnes et que les types de "
                         "données (ex: 'term' en nombre ou en texte) correspondent.")