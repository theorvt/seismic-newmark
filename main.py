import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from math import * 
import pandas as pd
from scipy.interpolate import interp1d

# Réduire les marges
st.set_page_config(layout="wide")

# Titre
st.markdown("Seismic simulation - Newmark method")
 
# Lien vers GeoNet
st.info("You can download seismic data from [GeoNet Strong Motion Database](https://data.geonet.org.nz/seismic-products/strong-motion/). Choose your station and event, then upload the `.csv` or `.txt` file below.")

# Barre latérale
st.sidebar.title("System settings ") 

 
# Paramètres modifiables
M = st.sidebar.slider("M : Mass (kg)", 1.0, 500.0, 80.0, step=1.0)
K = st.sidebar.slider("K : Stiffness (N/m)", 0.0, 50000.0, 10000.0, step=100.0)
zeta = st.sidebar.slider("zeta : Damping rate (%)", 0.0, 200.0, 5.0, step=1.0)

dt = st.sidebar.number_input("dt : Time step (s)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.4f")
F1 = st.sidebar.number_input("F1 : Amplitude coefficient (N)", min_value=0.0, max_value=10000.0, value=1.0, step=1.0, format="%.4f")

d0 = st.sidebar.slider("d : Initial movement (m)", 0.0, 0.5, 0.0, step=0.01)
v0 = st.sidebar.slider("v : Initial velocity (m/s)", 0.0, 1.0, 0.0, step=0.01)

scale = st.sidebar.radio("Period axis scale for response spectra", ("linear", "log"))

mu = st.sidebar.slider("μ : Friction coefficient", 0.0, 2.0, 0.01, step=0.01)
#K3 = st.sidebar.slider("K3 : Non-linear stiffness", 0.0, 1000000000.0, 100000.0, step=100.0)
log_K3 = st.sidebar.slider("log₁₀(K3)", 3, 9, 5, step=1)
K3 = 10 ** log_K3
st.sidebar.markdown(f"K3 : Non-linear stiffness = **{K3:.1e} N/m³**")

v_eps = 0.01  # petite valeur pour régularisation
N_force = M * 9.81  # force normale supposée


# Upload du fichier CSV ou Excel 
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"]) 

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                # Lecture auto avec détection de séparateur
                df = pd.read_csv(uploaded_file, skiprows=8, engine="python", sep=None)
            except Exception as e: 
                st.error(f"Error : CSV read failure : {e}")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file, skiprows=8)

        # Normalisation des noms 
        def normalize(col):
            return str(col).strip().lower().replace(" ", "").replace("\xa0", "").replace("_", "")

        col_map = {normalize(col): col for col in df.columns}
 
        # Récupérer les vrais noms de colonnes via leur nom normalisé
        time_col = col_map.get("time")
        vertical_col = col_map.get("vertical")
        h1_col = col_map.get("horizontal1")
        h2_col = col_map.get("horizontal2")
        
        component_options = []
        if vertical_col: component_options.append("Vertical")
        if h1_col: component_options.append("Horizontal 1")
        if h2_col: component_options.append("Horizontal 2")
        
        # Sécurité : vérifier qu’au moins une composante est disponible
        if not component_options:
            st.error("No vertical or horizontal components detected in the file.")
            st.stop()

        # Choix de la composante
        selected_component = st.sidebar.selectbox("Choose component", component_options)

        # Récupérer les données selon le choix
        if selected_component == "Vertical":
            acc_col = vertical_col
        elif selected_component == "Horizontal 1":
            acc_col = h1_col
        elif selected_component == "Horizontal 2":
            acc_col = h2_col
        else:
            st.error("Selected component not found.")
            st.stop()
 
        if time_col is None or acc_col is None:
            st.error("Columns 'Time' and 'Vertical' not detected.")
            st.stop()

        # Conversion du temps depuis format ISO 8601 en secondes écoulées
        if df[time_col].dtype == object:
            try:
                # Convertir en datetime
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

                # Vérifier qu'aucune conversion n'a échoué
                if df[time_col].isnull().any():
                    st.error("Impossible datetime conversion for certain tenses (invalid values).")
                    st.stop()

                # Calculer les secondes écoulées depuis le premier timestamp
                time_zero = df[time_col].iloc[0]
                df[time_col] = (df[time_col] - time_zero).dt.total_seconds()

            except Exception as e:
                st.error(f"Conversion of 'Time' impossible : {e}")
                st.stop()

        # Extraction des valeurs numériques
        time_data = pd.to_numeric(df[time_col], errors='coerce').values
        acc_data = pd.to_numeric(df[acc_col], errors='coerce').values

        # Filtrer les NaN
        valid = ~np.isnan(time_data) & ~np.isnan(acc_data)
        time_data = time_data[valid]
        acc_data = acc_data[valid]

    except Exception as e:
        st.error(f"Error : {e}")
        st.stop()


else:
    df = pd.read_csv('donnee_seisme_site_web.csv', sep=';')
    # Définition des colonnes des temps et des accélérations et conversion de la première colonne en float (temps)
    time_data = pd.to_numeric(df.iloc[:, 4], errors='coerce').values  # df.iloc[:, 4] Séléctionne la 5ième colonne du DataFrame
    acc_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  # pd.to_numeric(..., errors='coerce') : essaie de convertir chaque élément de cette colonne en nombre flottant (float).
    selected_component = "Vertical"
    # Si une valeur n’est pas convertible (ex. texte, cellule vide…), elle sera remplacée par NaN (Not a Number), grâce à errors='coerce'.
    # .values : transforme la série pandas en array NumPy pur, plus rapide à manipuler.



#Spectre de réponse de l'accélération des 10 étages

#Etage 10 
df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
time_data_etage_10 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
acc_data_etage_10 = pd.to_numeric(df.iloc[:, 1], errors='coerce').values

#Etage 9
df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
time_data_etage_9 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
acc_data_etage_9 = pd.to_numeric(df.iloc[:, 2], errors='coerce').values

#Etage 8
df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
time_data_etage_8 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
acc_data_etage_8 = pd.to_numeric(df.iloc[:, 3], errors='coerce').values

#Etage 7
df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
time_data_etage_7 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
acc_data_etage_7 = pd.to_numeric(df.iloc[:, 4], errors='coerce').values

#Etage 6
df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
time_data_etage_6 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
acc_data_etage_6 = pd.to_numeric(df.iloc[:, 5], errors='coerce').values

#Etage 5
df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
time_data_etage_5 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
acc_data_etage_5 = pd.to_numeric(df.iloc[:, 6], errors='coerce').values

#Etage 4
#df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
#time_data_etage_4 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
#acc_data_etage_4 = pd.to_numeric(df.iloc[:, 7], errors='coerce').values

#Etage 3
#df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
#time_data_etage_3 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
#acc_data_etage_3 = pd.to_numeric(df.iloc[:, 8], errors='coerce').values

#Etage 2
#df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
#time_data_etage_2 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values  
#acc_data_etage_2 = pd.to_numeric(df.iloc[:, 9], errors='coerce').values

#Etage 1
#df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
#time_data_etage_1 = pd.to_numeric(df.iloc[:, 0], errors='coerce').values   
#acc_data_etage_1 = pd.to_numeric(df.iloc[:, 10], errors='coerce').values




# Calcule automatique de la durée du fichier
T_max = float(np.nanmax(time_data)) if len(time_data) > 0 else 1000.0
T_default = round(T_max, 2)

# Slider de durée, ajusté dynamiquement
T = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max), value=T_default,
                      step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T)
    st.session_state["previous_T"] = T  # Mettre à jour la référence
    
   
    
    
# Version pour les etages
# Calcule automatique de la durée du fichier
T_max_etage_10 = float(np.nanmax(time_data_etage_10)) if len(time_data_etage_10) > 0 else 1000.0
T_default_etage_10 = round(T_max_etage_10, 2)

# Slider de durée, ajusté dynamiquement
T_etage_10 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_10), value=T_default_etage_10, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_10

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_10:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_10)
    st.session_state["previous_T"] = T_etage_10  # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_9 = float(np.nanmax(time_data_etage_9)) if len(time_data_etage_9) > 0 else 1000.0
T_default_etage_9 = round(T_max_etage_9, 2)

# Slider de durée, ajusté dynamiquement
T_etage_9 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_9), value=T_default_etage_9, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_9

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_9:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_9)
    st.session_state["previous_T"] = T_etage_9 # Mettre à jour la référence


# Calcule automatique de la durée du fichier
T_max_etage_8 = float(np.nanmax(time_data_etage_8)) if len(time_data_etage_8) > 0 else 1000.0
T_default_etage_8 = round(T_max_etage_8, 2)

# Slider de durée, ajusté dynamiquement
T_etage_8 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_8), value=T_default_etage_8, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_8

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_8:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_8)
    st.session_state["previous_T"] = T_etage_8  # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_7 = float(np.nanmax(time_data_etage_7)) if len(time_data_etage_7) > 0 else 1000.0
T_default_etage_7 = round(T_max_etage_7, 2)

# Slider de durée, ajusté dynamiquement
T_etage_7 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_7), value=T_default_etage_7, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_7

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_7:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_7)
    st.session_state["previous_T"] = T_etage_7 # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_6 = float(np.nanmax(time_data_etage_6)) if len(time_data_etage_6) > 0 else 1000.0
T_default_etage_6 = round(T_max_etage_6, 2)

# Slider de durée, ajusté dynamiquement
T_etage_6 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_6), value=T_default_etage_6, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_6

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_6:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_6)
    st.session_state["previous_T"] = T_etage_6  # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_5 = float(np.nanmax(time_data_etage_5)) if len(time_data_etage_5) > 0 else 1000.0
T_default_etage_5 = round(T_max_etage_5, 2)

# Slider de durée, ajusté dynamiquement
T_etage_5 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_5), value=T_default_etage_5, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_5

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_5:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_5)
    st.session_state["previous_T"] = T_etage_5  # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_4 = float(np.nanmax(time_data_etage_4)) if len(time_data_etage_4) > 0 else 1000.0
T_default_etage_4 = round(T_max_etage_4, 2)

# Slider de durée, ajusté dynamiquement
T_etage_4 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_4), value=T_default_etage_4, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_4

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_4:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_4)
    st.session_state["previous_T"] = T_etage_4  # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_3 = float(np.nanmax(time_data_etage_3)) if len(time_data_etage_3) > 0 else 1000.0
T_default_etage_3 = round(T_max_etage_3, 2)

# Slider de durée, ajusté dynamiquement
T_etage_3 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_3), value=T_default_etage_3, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_3

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_3:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_3)
    st.session_state["previous_T"] = T_etage_3  # Mettre à jour la référence
    
    
# Calcule automatique de la durée du fichier
T_max_etage_2 = float(np.nanmax(time_data_etage_2)) if len(time_data_etage_2) > 0 else 1000.0
T_default_etage_2 = round(T_max_etage_2, 2)

# Slider de durée, ajusté dynamiquement
T_etage_2 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_2), value=T_default_etage_2, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_2

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_2:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_2)
    st.session_state["previous_T"] = T_etage_2  # Mettre à jour la référence



# Calcule automatique de la durée du fichier
T_max_etage_1 = float(np.nanmax(time_data_etage_1)) if len(time_data_etage_1) > 0 else 1000.0
T_default_etage_1 = round(T_max_etage_1, 2)

# Slider de durée, ajusté dynamiquement
T_etage_1 = st.sidebar.slider("T : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage_1), value=T_default_etage_1, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T" not in st.session_state:
    st.session_state["previous_T"] = T_etage_1

if "time_range_slider" not in st.session_state or st.session_state["previous_T"] != T_etage_1:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider"] = (0.0, T_etage_1)
    st.session_state["previous_T"] = T_etage_1  # Mettre à jour la référence


    
params_key = (M, K, zeta, T, selected_component, d0, v0, dt, F1, scale, mu, K3)


# Définition du coefficent d'amortissement
C = (2 * (K * M) ** (1 / 2) * zeta) / 100
st.sidebar.markdown(f"C : Damping coefficient : **{C:.2f} Ns/m**")

# Filtrer les NaN
valid_indices = ~np.isnan(time_data) & ~np.isnan(acc_data)  # np.isnan(time_data) Cette fonction renvoie un tableau de booléens (True/False) de même taille que time_data.
# Elle contient True là où les valeurs sont NaN (Not a Number), c’est-à-dire des valeurs manquantes ou invalides.
# L’opérateur ~ est un NON logique (négation). Donc cette expression renvoie True pour les indices valides (non NaN) de time_data.
time_data = time_data[valid_indices]
acc_data = acc_data[valid_indices]



# Pour les etages
# Filtrer les NaN
valid_indices_etage_10 = ~np.isnan(time_data_etage_10) & ~np.isnan(acc_data_etage_10)  
time_data_etage_10 = time_data_etage_10[valid_indices_etage_10]
acc_data_etage_10 = acc_data_etage_10[valid_indices_etage_10]

valid_indices_etage_9 = ~np.isnan(time_data_etage_9) & ~np.isnan(acc_data_etage_9)  
time_data_etage_9 = time_data_etage_9[valid_indices_etage_9]
acc_data_etage_9 = acc_data_etage_10[valid_indices_etage_9]

valid_indices_etage_8 = ~np.isnan(time_data_etage_8) & ~np.isnan(acc_data_etage_8)  
time_data_etage_8 = time_data_etage_8[valid_indices_etage_8]
acc_data_etage_8 = acc_data_etage_10[valid_indices_etage_8]

valid_indices_etage_7 = ~np.isnan(time_data_etage_7) & ~np.isnan(acc_data_etage_7)  
time_data_etage_7 = time_data_etage_7[valid_indices_etage_7]
acc_data_etage_7 = acc_data_etage_10[valid_indices_etage_7]

valid_indices_etage_6 = ~np.isnan(time_data_etage_6) & ~np.isnan(acc_data_etage_6)  
time_data_etage_6 = time_data_etage_6[valid_indices_etage_6]
acc_data_etage_6 = acc_data_etage_10[valid_indices_etage_6]

valid_indices_etage_5 = ~np.isnan(time_data_etage_5) & ~np.isnan(acc_data_etage_5)  
time_data_etage_5 = time_data_etage_5[valid_indices_etage_5]
acc_data_etage_5 = acc_data_etage_10[valid_indices_etage_5]

valid_indices_etage_4 = ~np.isnan(time_data_etage_4) & ~np.isnan(acc_data_etage_4)  
time_data_etage_4 = time_data_etage_4[valid_indices_etage_4]
acc_data_etage_4 = acc_data_etage_10[valid_indices_etage_4]

valid_indices_etage_3 = ~np.isnan(time_data_etage_3) & ~np.isnan(acc_data_etage_3)  
time_data_etage_3 = time_data_etage_3[valid_indices_etage_3]
acc_data_etage_3 = acc_data_etage_10[valid_indices_etage_3]

valid_indices_etage_2 = ~np.isnan(time_data_etage_2) & ~np.isnan(acc_data_etage_2)  
time_data_etage_2 = time_data_etage_2[valid_indices_etage_2]
acc_data_etage_2 = acc_data_etage_10[valid_indices_etage_2]

valid_indices_etage_1 = ~np.isnan(time_data_etage_1) & ~np.isnan(acc_data_etage_1)  
time_data_etage_1 = time_data_etage_1[valid_indices_etage_1]
acc_data_etage_1 = acc_data_etage_1[valid_indices_etage_1]


# Définition d'un pas de temps adapté

# Paramètres Newmark (Ici on prend une accélération constante)
beta = 1 / 6
gamma = 1 / 2

# Définition de la pulsation propre du système
W = sqrt((K / M))
st.sidebar.markdown(f"ω₀ : Natural pulsation : **{W:.2f} rad/s**")
if W == 0:
    st.error("Error: ω₀ is zero (check M and K)")
    st.stop()
  
# Définition de la fréquence propre du système
f = W / (2 * pi)
st.sidebar.markdown(f"f : Natural frequency : **{f:.2f} Hz**")
if W == 0:
    st.error("Error: f is zero (check M and K)")
    st.stop()  
   
# Définition de la période propre du système
T0 = 1 / f
st.sidebar.markdown(f"T₀ : Natural frequency : **{T0:.2f} s**")
if W == 0:
    st.error("Error: T₀ is zero (check M and K)")
    st.stop()
    
#Vérification de la valeur de dt
dt1 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()


# Gestion des variables temporels
t = np.arange(0, T + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n = len(t)

# 🔧 MODIF : bornes du slider
time_min = 0.0
time_max = T


# Version pour les etages
#Vérification de la valeur de dt
dt1_etage_10 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_10:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_10 = np.arange(0, T_etage_10 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_10 = len(t_etage_10)

#  MODIF : bornes du slider
time_min_etage_10 = 0.0
time_max_etage_10 = T_etage_10


#Vérification de la valeur de dt
dt1_etage_9 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_9:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_9 = np.arange(0, T_etage_9 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_9 = len(t_etage_9)

#  MODIF : bornes du slider
time_min_etage_9 = 0.0
time_max_etage_9 = T_etage_9


#Vérification de la valeur de dt
dt1_etage_8 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_8:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_8 = np.arange(0, T_etage_8 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_8 = len(t_etage_8)

#  MODIF : bornes du slider
time_min_etage_8 = 0.0
time_max_etage_8 = T_etage_8


#Vérification de la valeur de dt
dt1_etage_7 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_7:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_7 = np.arange(0, T_etage_7 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_7 = len(t_etage_7)

#  MODIF : bornes du slider
time_min_etage_7 = 0.0
time_max_etage_7 = T_etage_7


#Vérification de la valeur de dt
dt1_etage_6 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_6:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_6 = np.arange(0, T_etage_6 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_6 = len(t_etage_6)

#  MODIF : bornes du slider
time_min_etage_6 = 0.0
time_max_etage_6 = T_etage_6


#Vérification de la valeur de dt
dt1_etage_5 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_5:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_5 = np.arange(0, T_etage_5 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_5 = len(t_etage_5)

#  MODIF : bornes du slider
time_min_etage_5 = 0.0
time_max_etage_5 = T_etage_5


#Vérification de la valeur de dt
dt1_etage_4 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_4:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_4 = np.arange(0, T_etage_4 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_4 = len(t_etage_4)

#  MODIF : bornes du slider
time_min_etage_4 = 0.0
time_max_etage_4 = T_etage_4


#Vérification de la valeur de dt
dt1_etage_3 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_3:
    st.error(f"⚠️ Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_3 = np.arange(0, T_etage_3 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_3 = len(t_etage_3)

#  MODIF : bornes du slider
time_min_etage_3 = 0.0
time_max_etage_3 = T_etage_3


#Vérification de la valeur de dt
dt1_etage_2 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_2:
    st.error(f"⚠️ Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_2 = np.arange(0, T_etage_2 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_2 = len(t_etage_2)

#  MODIF : bornes du slider
time_min_etage_2 = 0.0
time_max_etage_2 = T_etage_2


#Vérification de la valeur de dt
dt1_etage_1 = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage_1:
    st.error(f"⚠️ Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage_1 = np.arange(0, T_etage_1 + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage_1 = len(t_etage_1)

#  MODIF : bornes du slider
time_min_etage_1 = 0.0
time_max_etage_1 = T_etage_1



#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min, time_max)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min, time_max)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min, old_max = st.session_state["time_range_slider"]
    new_min = min(old_min, T)
    new_max = min(old_max, T)
    st.session_state["time_range_slider"] = (new_min, new_max)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min, time_max, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    
  
    
# Version pour les etages
#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_10, time_max_etage_10)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_10, time_max_etage_10)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_10, old_max_etage_10 = st.session_state["time_range_slider"]
    new_min_etage_10 = min(old_min_etage_10, T)
    new_max_etage_10 = min(old_max_etage_10, T)
    st.session_state["time_range_slider"] = (new_min_etage_10, new_max_etage_10)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_10, time_max_etage_10, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_10) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_10 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    
    
#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_9, time_max_etage_9)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_9, time_max_etage_9)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_9, old_max_etage_9 = st.session_state["time_range_slider"]
    new_min_etage_9 = min(old_min_etage_9, T)
    new_max_etage_9 = min(old_max_etage_9, T)
    st.session_state["time_range_slider"] = (new_min_etage_9, new_max_etage_9)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_9, time_max_etage_9, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_9) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_10 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    

#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_8, time_max_etage_8)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_8, time_max_etage_8)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_8, old_max_etage_8 = st.session_state["time_range_slider"]
    new_min_etage_8 = min(old_min_etage_8, T)
    new_max_etage_8 = min(old_max_etage_8, T)
    st.session_state["time_range_slider"] = (new_min_etage_8, new_max_etage_8)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_8, time_max_etage_8, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_8) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_8 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()


#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_7, time_max_etage_7)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_7, time_max_etage_7)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_7, old_max_etage_7 = st.session_state["time_range_slider"]
    new_min_etage_7 = min(old_min_etage_7, T)
    new_max_etage_7 = min(old_max_etage_7, T)
    st.session_state["time_range_slider"] = (new_min_etage_7, new_max_etage_7)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_7, time_max_etage_7, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_7) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_7 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()


#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_6, time_max_etage_6)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_6, time_max_etage_6)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_6, old_max_etage_6 = st.session_state["time_range_slider"]
    new_min_etage_6 = min(old_min_etage_6, T)
    new_max_etage_6 = min(old_max_etage_6, T)
    st.session_state["time_range_slider"] = (new_min_etage_6, new_max_etage_6)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_6, time_max_etage_6, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_6) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_6 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    
    
#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_5, time_max_etage_5)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_5, time_max_etage_5)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_5, old_max_etage_5 = st.session_state["time_range_slider"]
    new_min_etage_5 = min(old_min_etage_5, T)
    new_max_etage_5 = min(old_max_etage_5, T)
    st.session_state["time_range_slider"] = (new_min_etage_5, new_max_etage_5)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_5, time_max_etage_5, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_5) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_5 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()


#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_4, time_max_etage_4)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_4, time_max_etage_4)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_4, old_max_etage_4 = st.session_state["time_range_slider"]
    new_min_etage_4 = min(old_min_etage_4, T)
    new_max_etage_4 = min(old_max_etage_4, T)
    st.session_state["time_range_slider"] = (new_min_etage_4, new_max_etage_4)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_4, time_max_etage_4, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_4) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_4 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    
    
#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_3, time_max_etage_3)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_3, time_max_etage_3)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_3, old_max_etage_3 = st.session_state["time_range_slider"]
    new_min_etage_3 = min(old_min_etage_3, T)
    new_max_etage_3 = min(old_max_etage_3, T)
    st.session_state["time_range_slider"] = (new_min_etage_3, new_max_etage_3)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_3, time_max_etage_3, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_3) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_3 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    
    
#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_2, time_max_etage_2)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_2, time_max_etage_2)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_2, old_max_etage_2 = st.session_state["time_range_slider"]
    new_min_etage_2 = min(old_min_etage_2, T)
    new_max_etage_2 = min(old_max_etage_2, T)
    st.session_state["time_range_slider"] = (new_min_etage_2, new_max_etage_2)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_2, time_max_etage_2, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_2) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_2 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()
    
    
#  MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_1, time_max_etage_1)
    st.session_state["previous_T"] = T

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min_etage_1, time_max_etage_1)
    st.session_state["previous_T"] = T

elif st.session_state["previous_T"] != T:
    old_min_etage_1, old_max_etage_1 = st.session_state["time_range_slider"]
    new_min_etage_1 = min(old_min_etage_1, T)
    new_max_etage_1 = min(old_max_etage_1, T)
    st.session_state["time_range_slider"] = (new_min_etage_1, new_max_etage_1)
    st.session_state["previous_T"] = T

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min_etage_1, time_max_etage_1, key="time_range_slider",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage_1) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage_1 == 0:
    st.error("Error : The array of moments t is empty (Check T)")
    st.stop()





# Exécution unique de la simulation si les paramètres ont changé
if "results" not in st.session_state or st.session_state.get("last_params") != params_key:
    # Interpolation linéaire
    acc_interp = interp1d(time_data, acc_data, kind='linear', fill_value='extrapolate')
    accel = acc_interp(t)
    
    F = - F1 * M * accel
    F_friction = np.zeros_like(F)
    
    # Initialisation des réponses
    d = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    
    d_friction = np.zeros(n)
    v_friction = np.zeros(n)
    a_friction = np.zeros(n)
    
    d_non_lineaire = np.zeros(n)
    v_non_lineaire = np.zeros(n)
    a_non_lineaire = np.zeros(n)
    
    d_non_lineaire_friction = np.zeros(n)
    v_non_lineaire_friction = np.zeros(n)
    a_non_lineaire_friction = np.zeros(n)

    #Conditions initiales - Friction 
    friction = mu * N_force * (2 / np.pi) * np.arctan(v_friction[0] / v_eps)
    F_friction[0] = F[0] - friction

    # Conditions initiales - Modéle linéaire
    d[0] = d0
    v[0] = v0
    a[0] = (F[0] - C * v[0] - K * d[0]) / M
    
    # Conditions initiales - Modéle Avec friction
    d_friction[0] = d0
    v_friction[0] = v0
    a_friction[0] = (F_friction[0] - C * v_friction[0] - K * d_friction[0]) / M
    
    # Conditions initiales - Modèle non-linéaire
    d_non_lineaire[0] = d0
    v_non_lineaire[0] = v0
    a_non_lineaire[0] = (F[0] - C * v_non_lineaire[0] - K * d_non_lineaire[0] - K3 * d_non_lineaire[0]**3) / M
    
    # Conditions initiales - Modèle avec friction non-linéaire
    d_non_lineaire_friction[0] = d0
    v_non_lineaire_friction[0] = v0
    a_non_lineaire_friction[0] = (F_friction[0] - C * v_non_lineaire_friction[0] - K * d_non_lineaire_friction[0] - K3 * d_non_lineaire_friction[0]**3) / M
    
    
    # === Newton-Raphson + Newmark ===
    tol = 1e-6
    max_iter = 20
    

    # Pré-calculs
    B = M + K * beta * dt ** 2 + C * gamma * dt
    if B == 0:
        st.error("Error: Denominator B is zero. Try adjusting M, K, damping zeta, or time step dt.")
        st.stop()


    # Modèle linéaire
    for i in range(n - 1):
        # Prédictions Newmark
        P = v[i] + (1 - gamma) * dt * a[i]
        H = d[i] + dt * v[i] + (0.5 - beta) * dt**2 * a[i]

        # Mettre à jour les états
        a[i+1] = (F[i+1] - K * H - C * P) / B
        v[i+1] = P + gamma * dt * a[i+1]
        d[i+1] = H + beta * dt**2 * a[i+1]
     
         
    # Modèle linéaire avec friction
    for i in range(n - 1): 
        # Friction régulière (approximation continue)
        friction = mu * N_force * (2 / np.pi) * np.arctan(v_friction[i] / v_eps)

        # Force totale (avec frottement)
        F_friction[i+1] = F[i+1] - friction
         
        # Mettre à jour les états
        P_friction = v_friction[i] + (1 - gamma) * dt * a_friction[i]
        H_friction = d_friction[i] + dt * v_friction[i] + (0.5 - beta) * dt**2 * a_friction[i]
        
        a_friction[i + 1] = (F_friction[i + 1] - K * H_friction - C * P_friction) / B 
        v_friction[i + 1] = P_friction + gamma * dt * a_friction[i + 1] 
        d_friction[i + 1] = H_friction + beta * dt ** 2 * a_friction[i + 1] 
        
    
    # Modèle non-linéaire
    for i in range(n - 1):
        # Prédiction
        H_non_lineaire = d_non_lineaire[i] + dt * v_non_lineaire[i] + (0.5 - beta) * dt ** 2 * a_non_lineaire[i]
        P_non_lineaire = v_non_lineaire[i] + (1 - gamma) * dt * a_non_lineaire[i]

        d_guess = d_non_lineaire[i]
        
        for it in range(max_iter):
            a_guess = (d_guess - H_non_lineaire) / (beta * dt**2)
            v_guess = P_non_lineaire + gamma * dt * a_guess

            # Résidu
            R_non_lineaire = M * a_guess + C * v_guess + K * d_guess + K3 * d_guess**3 - F[i+1]

            # Dérivée du résidu
            dR_non_lineaire_dd = (M / (beta * dt**2) + gamma * dt * C / (beta * dt**2) + K + 3 * K3 * d_guess**2)
            
            delta_d = -R_non_lineaire / dR_non_lineaire_dd
            d_guess += delta_d

            if abs(delta_d) < tol:
               break
           
        else:
            print(f"Newton-Raphson did not converge at step {i+1}")
        
        # Mise à jour des états
        d_non_lineaire[i+1] = d_guess
        a_non_lineaire[i+1] = (d_non_lineaire[i+1] - H_non_lineaire) / (beta * dt**2)
        v_non_lineaire[i+1] = P_non_lineaire + gamma * dt * a_non_lineaire[i+1]
        
        
    # Modèle non-linéaire - avec friction
    for i in range(n - 1):
        # Prédiction
        H_non_lineaire_friction = d_non_lineaire_friction[i] + dt * v_non_lineaire_friction[i] + (0.5 - beta) * dt ** 2 * a_non_lineaire_friction[i]
        P_non_lineaire_friction = v_non_lineaire_friction[i] + (1 - gamma) * dt * a_non_lineaire_friction[i]

        d_guess_friction = d_non_lineaire_friction[i]
        
        for it in range(max_iter):
            a_guess_friction = (d_guess_friction - H_non_lineaire_friction) / (beta * dt**2)
            v_guess_friction = P_non_lineaire_friction + gamma * dt * a_guess_friction

            # Friction régulière (approximation continue)
            friction = mu * N_force * (2 / np.pi) * np.arctan(v_guess_friction / v_eps) 

            # Force totale (avec frottement)
            F_friction[i+1] = F[i+1] - friction

            # Résidu
            R_non_lineaire_friction = M * a_guess_friction + C * v_guess_friction + K * d_guess_friction + K3 * d_guess_friction **3 - F_friction[i+1]

            # Dérivée du résidu
            dR_non_lineaire_friction_dd = (M / (beta * dt**2) + gamma * dt * C / (beta * dt**2) + K + 3 * K3 * d_guess_friction ** 2)
            
            d_arctan = (2 / np.pi) * 1 / (1 + (v_guess_friction / v_eps)**2) / v_eps
            dR_non_lineaire_friction_dd += C * gamma * dt * mu * N_force * d_arctan / (beta * dt**2)

            delta_d_friction = -R_non_lineaire_friction / dR_non_lineaire_friction_dd
            d_guess_friction += delta_d_friction

            if abs(delta_d_friction) < tol:
               break
           
        else:
            print(f"Newton-Raphson did not converge at step {i+1}")
        
        # Mise à jour des états
        d_non_lineaire_friction[i+1] = d_guess_friction
        a_non_lineaire_friction[i+1] = (d_non_lineaire_friction[i+1] - H_non_lineaire_friction) / (beta * dt**2)
        v_non_lineaire_friction[i+1] = P_non_lineaire_friction + gamma * dt * a_non_lineaire_friction[i+1]
        
        
    # Calcul du spectre de Fourrier
    T0_list = np.linspace(0.02, 20, 250)
    
    #f_list = 1 / T0_list  # fréquence en Hz
    
    Sd, Sv, Sa = [], [], []
    
    for T0_i in T0_list: 
        ω_i = 2 * pi / T0_i
        K_i = M * ω_i**2
        C_i = 2 * M * ω_i * zeta / 100  # ζ en %

        Fsp = -M * accel  # acc = accélération au sol interpolée sur t
        
        # Initialisation
        dsp, vsp, asp = np.zeros(n), np.zeros(n), np.zeros(n)
        asp[0] = (Fsp[0] - C_i * vsp[0] - K_i * dsp[0]) / M

        # Newmark classique (β = 1/6, γ = 1/2)
        B = M + K_i * beta * dt**2 + C_i * gamma * dt
    
        for i in range(n-1):
            P = vsp[i] + (1 - gamma)*dt * asp[i]
            H = dsp[i] + dt * vsp[i] + (0.5 - beta)*dt**2 * asp[i]
            
            asp[i+1] = (Fsp[i+1] - K_i * H - C_i * P) / B
            vsp[i+1] = P + gamma*dt * asp[i+1]
            dsp[i+1] = H + beta*dt**2 * asp[i+1]

        # Stocker les maxima
        Sd.append(np.max(np.abs(dsp)))
        Sv.append(np.max(np.abs(vsp)))
        Sa.append(np.max(np.abs(asp)))
        
        
        
    #Version avec les etages
    # Interpolation linéaire
    acc_interp_etage_10 = interp1d(time_data_etage_10, acc_data_etage_10, kind='linear', fill_value='extrapolate')
    accel_etage_10 = acc_interp_etage_10(t)
    
    # Calcul du spectre de Fourrier
    T0_list_etage_10 = np.linspace(0.02, 20, 250)
    
    #f_list = 1 / T0_list  # fréquence en Hz
    
    Sa_etage_10 = []
    
    for T0_i_etage_10 in T0_list_etage_10: 
        ω_i_etage_10 = 2 * pi / T0_i_etage_10
        K_i_etage_10 = M * ω_i_etage_10**2
        C_i_etage_10 = 2 * M * ω_i_etage_10 * zeta / 100  # ζ en %

        Fsp_etage_10 = -M * accel_etage_10  # acc = accélération au sol interpolée sur t
        
        # Initialisation
        dsp_etage_10, vsp_etage_10, asp_etage_10 = np.zeros(n), np.zeros(n), np.zeros(n)
        asp_etage_10[0] = (Fsp_etage_10[0] - C_i_etage_10 * vsp_etage_10[0] - K_i_etage_10 * dsp_etage_10[0]) / M

        # Newmark classique (β = 1/6, γ = 1/2)
        B = M + K_i_etage_10 * beta * dt**2 + C_i_etage_10 * gamma * dt
    
        for i in range(n_etage_10-1):
            P = vsp_etage_10[i] + (1 - gamma)*dt * asp_etage_10[i]
            H = dsp_etage_10[i] + dt * vsp_etage_10[i] + (0.5 - beta)*dt**2 * asp_etage_10[i]
            
            asp_etage_10[i+1] = (Fsp_etage_10[i+1] - K_i_etage_10 * H - C_i_etage_10 * P) / B
            vsp_etage_10[i+1] = P + gamma*dt * asp_etage_10[i+1]
            dsp_etage_10[i+1] = H + beta*dt**2 * asp_etage_10[i+1]

        # Stocker les maxima
        Sa_etage_10.append(np.max(np.abs(asp_etage_10)))
        
        
        # Interpolation linéaire
        acc_interp_etage_1 = interp1d(time_data_etage_1, acc_data_etage_1, kind='linear', fill_value='extrapolate')
        accel_etage_1 = acc_interp_etage_1(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_1 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_1 = []
        
        for T0_i_etage_1 in T0_list_etage_1: 
            ω_i_etage_1 = 2 * pi / T0_i_etage_1
            K_i_etage_1 = M * ω_i_etage_1**2
            C_i_etage_1 = 2 * M * ω_i_etage_1 * zeta / 100  # ζ en %

            Fsp_etage_1 = -M * accel_etage_1  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_1, vsp_etage_1, asp_etage_1 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_1[0] = (Fsp_etage_1[0] - C_i_etage_1 * vsp_etage_1[0] - K_i_etage_1 * dsp_etage_1[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_1 * beta * dt**2 + C_i_etage_1 * gamma * dt
        
            for i in range(n_etage_1-1):
                P = vsp_etage_1[i] + (1 - gamma)*dt * asp_etage_1[i]
                H = dsp_etage_1[i] + dt * vsp_etage_1[i] + (0.5 - beta)*dt**2 * asp_etage_1[i]
                
                asp_etage_1[i+1] = (Fsp_etage_1[i+1] - K_i_etage_1 * H - C_i_etage_1 * P) / B
                vsp_etage_1[i+1] = P + gamma*dt * asp_etage_1[i+1]
                dsp_etage_1[i+1] = H + beta*dt**2 * asp_etage_1[i+1]

            # Stocker les maxima
            Sa_etage_1.append(np.max(np.abs(asp_etage_1)))
            
        
        # Interpolation linéaire
        acc_interp_etage_2 = interp1d(time_data_etage_2, acc_data_etage_2, kind='linear', fill_value='extrapolate')
        accel_etage_2 = acc_interp_etage_2(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_2 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_2 = []
        
        for T0_i_etage_2 in T0_list_etage_2: 
            ω_i_etage_2 = 2 * pi / T0_i_etage_2
            K_i_etage_2 = M * ω_i_etage_2**2
            C_i_etage_2 = 2 * M * ω_i_etage_2 * zeta / 100  # ζ en %

            Fsp_etage_2 = -M * accel_etage_2  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_2, vsp_etage_2, asp_etage_2 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_2[0] = (Fsp_etage_2[0] - C_i_etage_2 * vsp_etage_2[0] - K_i_etage_2 * dsp_etage_2[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_2 * beta * dt**2 + C_i_etage_2 * gamma * dt
        
            for i in range(n_etage_2-1):
                P = vsp_etage_2[i] + (1 - gamma)*dt * asp_etage_2[i]
                H = dsp_etage_2[i] + dt * vsp_etage_2[i] + (0.5 - beta)*dt**2 * asp_etage_2[i]
                
                asp_etage_2[i+1] = (Fsp_etage_2[i+1] - K_i_etage_2 * H - C_i_etage_2 * P) / B
                vsp_etage_2[i+1] = P + gamma*dt * asp_etage_2[i+1]
                dsp_etage_2[i+1] = H + beta*dt**2 * asp_etage_2[i+1]

            # Stocker les maxima
            Sa_etage_2.append(np.max(np.abs(asp_etage_2)))
        
        
        # Interpolation linéaire
        acc_interp_etage_3 = interp1d(time_data_etage_3, acc_data_etage_3, kind='linear', fill_value='extrapolate')
        accel_etage_3 = acc_interp_etage_3(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_3 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_3 = []
        
        for T0_i_etage_3 in T0_list_etage_3: 
            ω_i_etage_3 = 2 * pi / T0_i_etage_3
            K_i_etage_3 = M * ω_i_etage_3**2
            C_i_etage_3 = 2 * M * ω_i_etage_3 * zeta / 100  # ζ en %

            Fsp_etage_3 = -M * accel_etage_3  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_3, vsp_etage_3, asp_etage_3 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_3[0] = (Fsp_etage_3[0] - C_i_etage_3 * vsp_etage_3[0] - K_i_etage_3 * dsp_etage_3[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_3 * beta * dt**2 + C_i_etage_3 * gamma * dt
        
            for i in range(n_etage_3-1):
                P = vsp_etage_3[i] + (1 - gamma)*dt * asp_etage_3[i]
                H = dsp_etage_3[i] + dt * vsp_etage_3[i] + (0.5 - beta)*dt**2 * asp_etage_3[i]
                
                asp_etage_3[i+1] = (Fsp_etage_3[i+1] - K_i_etage_3 * H - C_i_etage_3 * P) / B
                vsp_etage_3[i+1] = P + gamma*dt * asp_etage_3[i+1]
                dsp_etage_3[i+1] = H + beta*dt**2 * asp_etage_3[i+1]

            # Stocker les maxima
            Sa_etage_3.append(np.max(np.abs(asp_etage_3)))
            
            
            
        # Interpolation linéaire
        acc_interp_etage_4 = interp1d(time_data_etage_4, acc_data_etage_4, kind='linear', fill_value='extrapolate')
        accel_etage_4 = acc_interp_etage_4(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_4 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_4 = []
        
        for T0_i_etage_4 in T0_list_etage_4: 
            ω_i_etage_4 = 2 * pi / T0_i_etage_4
            K_i_etage_4 = M * ω_i_etage_4**2
            C_i_etage_4 = 2 * M * ω_i_etage_4 * zeta / 100  # ζ en %

            Fsp_etage_4 = -M * accel_etage_4  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_4, vsp_etage_4, asp_etage_4 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_4[0] = (Fsp_etage_4[0] - C_i_etage_4 * vsp_etage_4[0] - K_i_etage_4 * dsp_etage_4[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_4 * beta * dt**2 + C_i_etage_4 * gamma * dt
        
            for i in range(n_etage_4-1):
                P = vsp_etage_4[i] + (1 - gamma)*dt * asp_etage_4[i]
                H = dsp_etage_4[i] + dt * vsp_etage_4[i] + (0.5 - beta)*dt**2 * asp_etage_4[i]
                
                asp_etage_4[i+1] = (Fsp_etage_4[i+1] - K_i_etage_4 * H - C_i_etage_4 * P) / B
                vsp_etage_4[i+1] = P + gamma*dt * asp_etage_4[i+1]
                dsp_etage_4[i+1] = H + beta*dt**2 * asp_etage_4[i+1]

            # Stocker les maxima
            Sa_etage_4.append(np.max(np.abs(asp_etage_4)))
            
            
        # Interpolation linéaire
        acc_interp_etage_5 = interp1d(time_data_etage_5, acc_data_etage_5, kind='linear', fill_value='extrapolate')
        accel_etage_5 = acc_interp_etage_5(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_5 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_5 = []
        
        for T0_i_etage_5 in T0_list_etage_5: 
            ω_i_etage_5 = 2 * pi / T0_i_etage_5
            K_i_etage_5 = M * ω_i_etage_5**2
            C_i_etage_5 = 2 * M * ω_i_etage_5 * zeta / 100  # ζ en %

            Fsp_etage_5 = -M * accel_etage_5  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_5, vsp_etage_5, asp_etage_5 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_5[0] = (Fsp_etage_5[0] - C_i_etage_5 * vsp_etage_5[0] - K_i_etage_5 * dsp_etage_5[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_5 * beta * dt**2 + C_i_etage_5 * gamma * dt
        
            for i in range(n_etage_5-1):
                P = vsp_etage_5[i] + (1 - gamma)*dt * asp_etage_5[i]
                H = dsp_etage_5[i] + dt * vsp_etage_5[i] + (0.5 - beta)*dt**2 * asp_etage_5[i]
                
                asp_etage_5[i+1] = (Fsp_etage_5[i+1] - K_i_etage_5 * H - C_i_etage_5 * P) / B
                vsp_etage_5[i+1] = P + gamma*dt * asp_etage_5[i+1]
                dsp_etage_5[i+1] = H + beta*dt**2 * asp_etage_5[i+1]

            # Stocker les maxima
            Sa_etage_5.append(np.max(np.abs(asp_etage_5)))
        
        
        # Interpolation linéaire
        acc_interp_etage_6 = interp1d(time_data_etage_6, acc_data_etage_6, kind='linear', fill_value='extrapolate')
        accel_etage_6 = acc_interp_etage_6(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_6 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_6 = []
        
        for T0_i_etage_6 in T0_list_etage_6: 
            ω_i_etage_6 = 2 * pi / T0_i_etage_6
            K_i_etage_6 = M * ω_i_etage_6**2
            C_i_etage_6 = 2 * M * ω_i_etage_6 * zeta / 100  # ζ en %

            Fsp_etage_6 = -M * accel_etage_6  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_6, vsp_etage_6, asp_etage_6 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_6[0] = (Fsp_etage_6[0] - C_i_etage_6 * vsp_etage_6[0] - K_i_etage_6 * dsp_etage_6[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_6 * beta * dt**2 + C_i_etage_6 * gamma * dt
        
            for i in range(n_etage_6-1):
                P = vsp_etage_6[i] + (1 - gamma)*dt * asp_etage_6[i]
                H = dsp_etage_6[i] + dt * vsp_etage_6[i] + (0.5 - beta)*dt**2 * asp_etage_6[i]
                
                asp_etage_6[i+1] = (Fsp_etage_6[i+1] - K_i_etage_6 * H - C_i_etage_6 * P) / B
                vsp_etage_6[i+1] = P + gamma*dt * asp_etage_6[i+1]
                dsp_etage_6[i+1] = H + beta*dt**2 * asp_etage_6[i+1]

            # Stocker les maxima
            Sa_etage_6.append(np.max(np.abs(asp_etage_6)))
            
            
        # Interpolation linéaire
        acc_interp_etage_7 = interp1d(time_data_etage_7, acc_data_etage_7, kind='linear', fill_value='extrapolate')
        accel_etage_7 = acc_interp_etage_7(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_7 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_7 = []
        
        for T0_i_etage_7 in T0_list_etage_7: 
            ω_i_etage_7 = 2 * pi / T0_i_etage_7
            K_i_etage_7 = M * ω_i_etage_7**2
            C_i_etage_7 = 2 * M * ω_i_etage_7 * zeta / 100  # ζ en %

            Fsp_etage_7 = -M * accel_etage_7  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_7, vsp_etage_7, asp_etage_7 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_7[0] = (Fsp_etage_7[0] - C_i_etage_7 * vsp_etage_7[0] - K_i_etage_7 * dsp_etage_7[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_7 * beta * dt**2 + C_i_etage_7 * gamma * dt
        
            for i in range(n_etage_7-1):
                P = vsp_etage_7[i] + (1 - gamma)*dt * asp_etage_7[i]
                H = dsp_etage_7[i] + dt * vsp_etage_7[i] + (0.5 - beta)*dt**2 * asp_etage_7[i]
                
                asp_etage_7[i+1] = (Fsp_etage_7[i+1] - K_i_etage_7 * H - C_i_etage_7 * P) / B
                vsp_etage_7[i+1] = P + gamma*dt * asp_etage_7[i+1]
                dsp_etage_7[i+1] = H + beta*dt**2 * asp_etage_7[i+1]

            # Stocker les maxima
            Sa_etage_7.append(np.max(np.abs(asp_etage_7)))
            
            
        # Interpolation linéaire
        acc_interp_etage_8 = interp1d(time_data_etage_8, acc_data_etage_8, kind='linear', fill_value='extrapolate')
        accel_etage_8 = acc_interp_etage_8(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_8 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_8 = []
        
        for T0_i_etage_8 in T0_list_etage_8: 
            ω_i_etage_8 = 2 * pi / T0_i_etage_8
            K_i_etage_8 = M * ω_i_etage_8**2
            C_i_etage_8 = 2 * M * ω_i_etage_8 * zeta / 100  # ζ en %

            Fsp_etage_8 = -M * accel_etage_8  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_8, vsp_etage_8, asp_etage_8 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_8[0] = (Fsp_etage_8[0] - C_i_etage_8 * vsp_etage_8[0] - K_i_etage_8 * dsp_etage_8[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_8 * beta * dt**2 + C_i_etage_8 * gamma * dt
        
            for i in range(n_etage_8-1):
                P = vsp_etage_8[i] + (1 - gamma)*dt * asp_etage_8[i]
                H = dsp_etage_8[i] + dt * vsp_etage_8[i] + (0.5 - beta)*dt**2 * asp_etage_8[i]
                
                asp_etage_8[i+1] = (Fsp_etage_8[i+1] - K_i_etage_8 * H - C_i_etage_8 * P) / B
                vsp_etage_8[i+1] = P + gamma*dt * asp_etage_8[i+1]
                dsp_etage_8[i+1] = H + beta*dt**2 * asp_etage_8[i+1]

            # Stocker les maxima
            Sa_etage_8.append(np.max(np.abs(asp_etage_8)))
            
            
        # Interpolation linéaire
        acc_interp_etage_9 = interp1d(time_data_etage_9, acc_data_etage_9, kind='linear', fill_value='extrapolate')
        accel_etage_9 = acc_interp_etage_9(t)
        
        # Calcul du spectre de Fourrier
        T0_list_etage_9 = np.linspace(0.02, 20, 250)
        
        #f_list = 1 / T0_list  # fréquence en Hz
        
        Sa_etage_9 = []
        
        for T0_i_etage_9 in T0_list_etage_9: 
            ω_i_etage_9 = 2 * pi / T0_i_etage_9
            K_i_etage_9 = M * ω_i_etage_9**2
            C_i_etage_9 = 2 * M * ω_i_etage_9 * zeta / 100  # ζ en %

            Fsp_etage_9 = -M * accel_etage_9  # acc = accélération au sol interpolée sur t
            
            # Initialisation
            dsp_etage_9, vsp_etage_9, asp_etage_9 = np.zeros(n), np.zeros(n), np.zeros(n)
            asp_etage_9[0] = (Fsp_etage_9[0] - C_i_etage_9 * vsp_etage_9[0] - K_i_etage_9 * dsp_etage_9[0]) / M

            # Newmark classique (β = 1/6, γ = 1/2)
            B = M + K_i_etage_9 * beta * dt**2 + C_i_etage_9 * gamma * dt
        
            for i in range(n_etage_9-1):
                P = vsp_etage_9[i] + (1 - gamma)*dt * asp_etage_8[i]
                H = dsp_etage_9[i] + dt * vsp_etage_9[i] + (0.5 - beta)*dt**2 * asp_etage_9[i]
                
                asp_etage_9[i+1] = (Fsp_etage_9[i+1] - K_i_etage_9 * H - C_i_etage_9 * P) / B
                vsp_etage_9[i+1] = P + gamma*dt * asp_etage_9[i+1]
                dsp_etage_9[i+1] = H + beta*dt**2 * asp_etage_9[i+1]

            # Stocker les maxima
            Sa_etage_9.append(np.max(np.abs(asp_etage_9)))
        
        
        
        
    # Sauvegarde des résultats
    st.session_state.results = {"t": t, "F": F, "d": d, "v": v, "a": a, "Sd": Sd, "Sv": Sv, "Sa": Sa, "T0_list": T0_list, "d_friction": d_friction, "v_friction": v_friction, "a_friction": a_friction, "d_non_lineaire": d_non_lineaire, "v_non_lineaire": v_non_lineaire, "a_non_lineaire": a_non_lineaire, "d_non_lineaire_friction": d_non_lineaire_friction, "v_non_lineaire_friction": v_non_lineaire_friction, "a_non_lineaire_friction": a_non_lineaire_friction, 
                                "Sa_etage_10": Sa_etage_10, "Sa_etage_9": Sa_etage_9, "Sa_etage_8": Sa_etage_8, "Sa_etage_7": Sa_etage_7, "Sa_etage_6": Sa_etage_6, "Sa_etage_5": Sa_etage_5, "Sa_etage_4": Sa_etage_4, "Sa_etage_3": Sa_etage_3, "Sa_etage_2": Sa_etage_2, "Sa_etage_1": Sa_etage_1,}
    st.session_state.last_params = params_key

# Récupération des résultats depuis session_state
t = st.session_state.results["t"]
F = st.session_state.results["F"]

d = st.session_state.results["d"]
v = st.session_state.results["v"]
a = st.session_state.results["a"]

d_friction = st.session_state.results["d_friction"]
v_friction = st.session_state.results["v_friction"]
a_friction = st.session_state.results["a_friction"]

d_non_lineaire = st.session_state.results["d_non_lineaire"]
v_non_lineaire = st.session_state.results["v_non_lineaire"]
a_non_lineaire = st.session_state.results["a_non_lineaire"]

d_non_lineaire_friction = st.session_state.results["d_non_lineaire_friction"]
v_non_lineaire_friction = st.session_state.results["v_non_lineaire_friction"]
a_non_lineaire_friction = st.session_state.results["a_non_lineaire_friction"]

T0_list = st.session_state.results["T0_list"]

Sd = st.session_state.results["Sd"]
Sv = st.session_state.results["Sv"]
Sa = st.session_state.results["Sa"]

Sa_etage_10 = st.session_state.results["Sa_etage_10"]
Sa_etage_9 = st.session_state.results["Sa_etage_9"]
Sa_etage_8 = st.session_state.results["Sa_etage_8"]
Sa_etage_7 = st.session_state.results["Sa_etage_7"]
Sa_etage_6 = st.session_state.results["Sa_etage_6"]
Sa_etage_5 = st.session_state.results["Sa_etage_5"]
Sa_etage_4 = st.session_state.results["Sa_etage_4"]
Sa_etage_3 = st.session_state.results["Sa_etage_3"]
Sa_etage_2 = st.session_state.results["Sa_etage_2"]
Sa_etage_1 = st.session_state.results["Sa_etage_1"]

# Indices correspondant à la plage de temps sélectionnée
mask = (t >= selected_range[0]) & (t <= selected_range[1])

# Filtrage des données
t = t[mask]
F = F[mask]
d = d[mask] 
v = v[mask]
a = a[mask] 

d_friction = d_friction[mask]
v_friction = v_friction[mask]
a_friction = a_friction[mask]

d_non_lineaire = d_non_lineaire[mask]
v_non_lineaire = v_non_lineaire[mask]
a_non_lineaire = a_non_lineaire[mask]

d_non_lineaire_friction = d_non_lineaire_friction[mask]
v_non_lineaire_friction = v_non_lineaire_friction[mask]
a_non_lineaire_friction = a_non_lineaire_friction[mask]
 

# Affichage

# Affichage d'un titre si l'utilisateur n'a pas encore uploadé de fichier
if uploaded_file is None:
    selected_component = "Vertical"
    st.markdown("### Example of simulation with default data")
    st.info("You are currently viewing a simulation example with predefined data. To use your own seismic data, import a CSV or Excel file at the top of the page.")



# Enregistrement du séisme et spectre de réponse
st.markdown("Earthquake input")

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, F, color="#0072CE")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Force(N)")
    ax.set_title(f"Ground acceleration - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    
with col2:
    fig, ax = plt.subplots()
    ax.plot(T0_list, Sa, color="#002B45")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Peak Acceleration")
    ax.set_title(f"Acceleration response spectrum - {selected_component}")
    ax.set_xscale(scale)
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    
with col3:
    fig, ax = plt.subplots()
    ax.plot(T0_list, Sd, color="#002B45")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Peak Displacement")
    ax.set_title(f"Displacement response spectrum  - {selected_component}")
    ax.set_xscale(scale)
    ax.grid()
    ax.legend()
    st.pyplot(fig)    
    
    
    
# Mode linéaire
st.markdown("SDOF Structural response - Linear Model")   

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, d, color="#002B45")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Displacement time history - Linear model - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(t, v, color="#009CA6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity time history - Linear model - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    ax.plot(t, a, color="#1C2D3F")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Acceleration time history - Linear model - {selected_component}") 
    ax.grid()
    ax.legend()
    st.pyplot(fig)
  
    

# Mode non-linéaire
st.markdown("SDOF Structural response - Non Linear Model")   

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, d_non_lineaire, color="#002B45")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Displacement time history - Non Linear model - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(t, v_non_lineaire, color="#009CA6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity time history - Non Linear model - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    ax.plot(t, a_non_lineaire, color="#1C2D3F")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Acceleration time history - Non Linear model - {selected_component}") 
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    
    
    
# Mode linéaire avec friction
st.markdown("SDOF Structural response - Linear Model with Friction")   

col1, col2, col3 = st.columns(3)  
   
with col1:
    fig, ax = plt.subplots()
    ax.plot(t, d_friction, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Displacement time history - Linear model with friction - {selected_component}")
    ax.grid()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(t, v_friction, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity time history - Linear model with friction - {selected_component}")
    ax.grid()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    ax.plot(t, a_friction, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Acceleration time history - Linear model with friction - {selected_component}")
    ax.grid()
    st.pyplot(fig)
    
    
    
# Mode non-linéaire avec friction
st.markdown("SDOF Structural response - Non Linear Model with Friction")   

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, d_non_lineaire_friction, color="#002B45")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Displacement time history - Non Linear model - Friction - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(t, v_non_lineaire_friction, color="#009CA6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity time history - Non Linear model - Friction - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    ax.plot(t, a_non_lineaire_friction, color="#1C2D3F")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Acceleration time history - Non Linear model - Friction - {selected_component}") 
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    
    
def raideur_non_lineaire(d_non_lineaire):
    F_raideur_non_lineaire = np.zeros(n)
    for i in range(n-1):
        F_raideur_non_lineaire[i] = K * d_non_lineaire[i] + K3 * d_non_lineaire[i] ** 3
    return F_raideur_non_lineaire

F_raideur_non_lineaire = raideur_non_lineaire(d_non_lineaire)

# Représentation graphique de la raideur non_linéaire
st.markdown("Non linear stiffness")   

fig, ax = plt.subplots()
ax.plot(d_non_lineaire, F_raideur_non_lineaire, color="#002B45")
ax.set_xlabel("Displacement (m)")
ax.set_ylabel("Stiffness force")
ax.set_title(f"Stiffness force - Non Linear model - {selected_component}")
ax.grid()
ax.legend()
st.pyplot(fig)


# Affichage du graphiques pour les étages

st.markdown("Te Puni building floor reaction")

fig, ax = plt.subplots()
ax.plot(T0_list_etage_10, Sa_etage_10, Sa_etage_9, Sa_etage_8, Sa_etage_7, Sa_etage_6, Sa_etage_5, Sa_etage_4, Sa_etage_3, Sa_etage_2, Sa_etage_1, color="#002B45")
ax.set_xlabel("Period (s)")
ax.set_ylabel("Peak Acceleration")
ax.set_title("Acceleration response spectrum per floor")
ax.set_xscale(scale)
ax.grid()
ax.legend()
st.pyplot(fig) 



output_df = pd.DataFrame(
    {"Time (s)": t, "Displacement (m)": d, "Velocity (m/s)": v, "Acceleration (m/s²)": a, "Force (N)": F})
csv = output_df.to_csv(index=False).encode('utf-8')
st.download_button("Download results as CSV", data=csv, file_name='newmark_results.csv', mime='text/csv')
