import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from math import sqrt, pi
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

dt = st.sidebar.number_input("dt : Time step (s)", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
F1 = st.sidebar.number_input("F1 : Amplitude coefficient (N)", min_value=0.0, max_value=10000.0, value=1.0, step=1.0, format="%.4f")

d0 = st.sidebar.slider("d : Initial movement (m)", 0.0, 0.5, 0.0, step=0.01)
v0 = st.sidebar.slider("v : Initial velocity (m/s)", 0.0, 1.0, 0.0, step=0.01)

scale = st.sidebar.radio("Period axis scale for response spectra", ("linear", "log"))

mu = st.sidebar.slider("μ : Friction coefficient", 0.0, 2.0, 0.01, step=0.01)
log_K3 = st.sidebar.slider("log₁₀(K3)", 3, 9, 5, step=1)
K3 = 10 ** log_K3
st.sidebar.markdown(f"K3 : Non-linear stiffness = **{K3:.1e} N/m³**")

# Liste des graphiques disponibles
options = ["Earthquake input", "Acceleration and Displacement spectrum", "SDOF Structural response - Linear Model","SDOF Structural response - Non Linear Model","SDOF Structural response - Linear Model with Friction",
           "SDOF Structural response - Non Linear Model with Friction","Te Puni building floor reaction", "Stiffness"]

n_floors = st.number_input("Enter the number of floor you want to consider", min_value=1, max_value=100, value=12, step=1)

# Sélection par l'utilisateur
graphique_choisi = st.selectbox("Which graph do you want to display ?", options)

v_eps = 0.01  # petite valeur pour régularisation
N_force = M * 9.81  # force normale supposée


# Upload du fichier CSV ou Excel 
uploaded_file = st.file_uploader("Upload a CSV or Excel file of an earthquake", type=["csv", "xls", "xlsx"])

# Upload du fichier CSV ou Excel 
uploaded_file_2 = st.file_uploader("Upload a CSV or Excel file of floor data", type=["csv", "xls", "xlsx"])


if uploaded_file_2 is not None:
   df_etage = pd.read_csv(uploaded_file_2, engine="python", sep=';')
   time_data_etage = pd.to_numeric(df_etage.iloc[:, 0], errors='coerce').values

   acc_data_etage = []
   for l in range(1, n_floors+1):
       acc_data_etage.append(pd.to_numeric(df_etage.iloc[:, l], errors='coerce').values)
    
   nb_etage = len(acc_data_etage) 
   
else:
    df_etage = pd.read_csv('donnee_10_etages.csv', sep=';')
    time_data_etage = pd.to_numeric(df_etage.iloc[:, 0], errors='coerce').values

    acc_data_etage = []
    for l in range(1, n_floors+1):
        acc_data_etage.append(pd.to_numeric(df_etage.iloc[:, l], errors='coerce').values)
     
    nb_etage = len(acc_data_etage)
    

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
T_max_etage = float(np.nanmax(time_data_etage)) if len(time_data_etage) > 0 else 1000.0
T_default_etage = round(T_max_etage, 2)

# Slider de durée, ajusté dynamiquement
T_etage = st.sidebar.slider("T_etage : Duration of the simulation (s)", min_value=0.01, max_value=float(T_max_etage), value=T_default_etage, step=0.01)

# Détection de changement de T (durée totale)
if "previous_T_etage" not in st.session_state:
    st.session_state["previous_T_etage"] = T_etage

if "time_range_slider_etage" not in st.session_state or st.session_state["previous_T_etage"] != T_etage:
    # Réinitialiser la plage au nouveau T
    st.session_state["time_range_slider_etage"] = (0.0, T_etage)
    st.session_state["previous_T_etage"] = T_etage  # Mettre à jour la référence
    

params_key = (M, K, zeta, T, selected_component, d0, v0, dt, F1, scale, mu, K3, graphique_choisi, n_floors)


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
acc_data_etage = []
for i in range(1, 12):  # Étages 1 à 10
    acc_data_etage.append(pd.to_numeric(df_etage.iloc[:, 13 - i], errors="coerce").values)

# Masque global
global_valid = ~np.isnan(time_data_etage)
for acc in acc_data_etage:
    global_valid &= ~np.isnan(acc)

time_data_etage = time_data_etage[global_valid]
acc_data_etage = [acc[global_valid] for acc in acc_data_etage]

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
dt1_etage = 2 / W * sqrt(1 / (1 - 2 * beta))  # Formule correcte
if dt > dt1_etage:
    st.error(f"Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()

# Gestion des variables temporels
t_etage = np.arange(0, T_etage + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n_etage = len(t_etage)

#  MODIF : bornes du slider
time_min_etage = 0.0
time_max_etage = T_etage


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
if "time_range_slider_etage" not in st.session_state or "previous_T_etage" not in st.session_state:
    st.session_state["time_range_slider_etage"] = (time_min_etage, time_max_etage)
    st.session_state["previous_T_etage"] = T_etage

#  MODIF : mise à jour du slider si T a changé
if "time_range_slider_etage" not in st.session_state:
    st.session_state["time_range_slider_etage"] = (time_min_etage, time_max_etage)
    st.session_state["previous_T_etage"] = T_etage

elif st.session_state["previous_T_etage"] != T_etage:
    old_min_etage, old_max_etage = st.session_state["time_range_slider_etage"]
    new_min_etage = min(old_min_etage, T_etage)
    new_max_etage = min(old_max_etage, T_etage)
    st.session_state["time_range_slider_etage"] = (new_min_etage, new_max_etage)
    st.session_state["previous_T_etage"] = T_etage

selected_range_etage = st.sidebar.slider("Te Puni - Select the time range to display (s)", time_min_etage, time_max_etage, key="time_range_slider_etage",step=1.0)

# Défintion des conditions de stabilité du programme
if len(time_data_etage) == 0:
    st.error("Error : The time data are empty after the filtration")
    st.stop()

if T_etage <= dt:
    st.error("Error : The total duration needs to be higher that the step time dt")
    st.stop()

if n_etage == 0:
    st.error("Error : The array of moments t is empty (Check T_etage)")
    st.stop()
    
    
# Calcul de l'accélération à partir de la modélisation
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


    # Affichage d'un titre si l'utilisateur n'a pas encore uploadé de fichier
    if uploaded_file is None:
        selected_component = "Vertical"
        st.markdown("### Example of simulation with default data")
        st.info("You are currently viewing a simulation example with predefined data. To use your own seismic data, import a CSV or Excel file at the top of the page.")


    # Modèle linéaire
    def newmark_lineaire():
        for i in range(n - 1):
            # Prédictions Newmark
            P = v[i] + (1 - gamma) * dt * a[i]
            H = d[i] + dt * v[i] + (0.5 - beta) * dt**2 * a[i]
    
            # Mettre à jour les états
            a[i+1] = (F[i+1] - K * H - C * P) / B
            v[i+1] = P + gamma * dt * a[i+1]
            d[i+1] = H + beta * dt**2 * a[i+1]
            
        return a, v, d
            
            
    # Modèle linéaire avec friction       
    def newmark_lineaire_friction():
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
            
        return a_friction, v_friction, d_friction
       
            
    # Modèle non-linéaire
    def newmark_non_lineaire():
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
        
        return a_non_lineaire, v_non_lineaire, d_non_lineaire
        
     
    # Modèle non-linéaire - avec friction
    def newmark_non_lineaire_friction():
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
            
        return a_non_lineaire_friction, v_non_lineaire_friction, d_non_lineaire_friction
    
    
    # Spectre acceleration et deplacement
    def newmark_spectrum():
        T0_list = np.linspace(0.02, 3, 50)
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
            
        return Sa, Sv, Sd
    
    
    # Spectre étages
    def newmark_spectrum_etage():
        #Version avec les etages 
        acc_interp_etage = []
        accel_etage = []
    
        for i in range(len(acc_data_etage)):
            f_interp = interp1d(time_data_etage, acc_data_etage[i], kind='linear', fill_value='extrapolate')
            acc_interp_etage.append(f_interp)   # stocke la fonction
            accel_etage.append(f_interp(t))     # applique la fonction tout de suite
               
        # Calcul du spectre de Fourrier
        T0_list_etage = np.linspace(0.01, 1, 50)
        
        # Spectre de réponse
        Sa_etage = [[] for m in range(nb_etage)]  # 12 étages
        
        for j, acc_j in enumerate(accel_etage):
            
            #acc_j = accel_etage[j]  # accélération interpolée de l’étage j
            
            for T0_i_etage in T0_list_etage: 
                
                ω_i = 2 * pi / T0_i_etage
                K_i = M * ω_i**2
                C_i = 2 * M * ω_i * zeta / 100  # ζ en %
                
                Fsp_etage = -M * acc_j
                
                # Initialisation
                dsp_etage, vsp_etage, asp_etage = np.zeros(n), np.zeros(n), np.zeros(n)
                asp_etage[0] = (Fsp_etage[0] - C_i * vsp_etage[0] - K_i * dsp_etage[0]) / M
    
                # Newmark classique (β = 1/6, γ = 1/2)
                B = M + K_i * beta * dt**2 + C_i * gamma * dt
            
                for i in range(len(t)-1):
                    P = vsp_etage[i] + (1 - gamma)*dt * asp_etage[i]
                    H = dsp_etage[i] + dt * vsp_etage[i] + (0.5 - beta)*dt**2 * asp_etage[i]
                    
                    asp_etage[i+1] = (Fsp_etage[i+1] - K_i * H - C_i * P) / B
                    vsp_etage[i+1] = P + gamma*dt * asp_etage[i+1]
                    dsp_etage[i+1] = H + beta*dt**2 * asp_etage[i+1]
    
                # Stocker les maxima
                Sa_etage[j].append(np.max(np.abs(asp_etage)))
                
        for j in range(nb_etage):
            if len(Sa_etage[j]) == 0:
              # rien calculé → on remplit par des zéros
              Sa_etage[j] = np.zeros(len(T0_list_etage))
            elif len(Sa_etage[j]) != len(T0_list_etage):
            # interpolation si besoin
                Sa_etage[j] = np.interp(np.linspace(0, 1, len(T0_list_etage)),np.linspace(0, 1, len(Sa_etage[j])),Sa_etage[j])
        
        return Sa_etage
    
    
    if graphique_choisi == "Earthquake input":
        
        st.markdown("Earthquake input")
        
        fig, ax = plt.subplots()
        ax.plot(t, F, color="#0072CE")
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Force(N)")
        ax.set_title(f"Ground acceleration - {selected_component}")
        ax.grid()
        st.pyplot(fig)
        

    elif graphique_choisi == "SDOF Structural response - Linear Model":
        
        a, v ,d = newmark_lineaire()
        
        # Sauvegarde des résultats
        st.session_state.results = {"t": t, "F": F, "d": d, "v": v, "a": a}
        st.session_state.last_params = params_key
        
        # Récupération des résultats depuis session_state
        t = st.session_state.results["t"]
        F = st.session_state.results["F"]

        d = st.session_state.results["d"]
        v = st.session_state.results["v"]
        a = st.session_state.results["a"]
        
        # Indices correspondant à la plage de temps sélectionnée
        mask = (t >= selected_range[0]) & (t <= selected_range[1])

        # Filtrage des données
        t = t[mask]
        F = F[mask]
        d = d[mask] 
        v = v[mask]
        a = a[mask] 
        
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
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            ax.plot(t, v, color="#009CA6")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity")
            ax.set_title(f"Velocity time history - Linear model - {selected_component}")
            ax.grid()
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots()
            ax.plot(t, a, color="#1C2D3F")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration")
            ax.set_title(f"Acceleration time history - Linear model - {selected_component}") 
            ax.grid()
            st.pyplot(fig)

        
        
    elif graphique_choisi == "SDOF Structural response - Linear Model with Friction":   
        
        a_friction, v_friction, d_friction = newmark_lineaire_friction()
        
        # Sauvegarde des résultats
        st.session_state.results = {"t": t, "F": F, "d_friction": d_friction, "v_friction": v_friction, "a_friction": a_friction}
        st.session_state.last_params = params_key
        
        # Récupération des résultats depuis session_state
        t = st.session_state.results["t"]
        F = st.session_state.results["F"]

        d_friction = st.session_state.results["d_friction"]
        v_friction = st.session_state.results["v_friction"]
        a_friction = st.session_state.results["a_friction"]
        
        # Indices correspondant à la plage de temps sélectionnée
        mask = (t >= selected_range[0]) & (t <= selected_range[1])

        # Filtrage des données
        t = t[mask]
        F = F[mask]
         
        d_friction = d_friction[mask]
        v_friction = v_friction[mask]
        a_friction = a_friction[mask]
        
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
        
        
    elif graphique_choisi == "SDOF Structural response - Non Linear Model":
        
        a_non_lineaire, v_non_lineaire, d_non_lineaire = newmark_non_lineaire()
        
        # Sauvegarde des résultats
        st.session_state.results = {"t": t, "F": F, "d_non_lineaire": d_non_lineaire, "v_non_lineaire": v_non_lineaire, "a_non_lineaire": a_non_lineaire}
        st.session_state.last_params = params_key
        
        # Récupération des résultats depuis session_state
        t = st.session_state.results["t"]
        F = st.session_state.results["F"]

        d_non_lineaire = st.session_state.results["d_non_lineaire"]
        v_non_lineaire = st.session_state.results["v_non_lineaire"]
        a_non_lineaire = st.session_state.results["a_non_lineaire"]
        
        # Indices correspondant à la plage de temps sélectionnée
        mask = (t >= selected_range[0]) & (t <= selected_range[1])

        # Filtrage des données
        t = t[mask]
        F = F[mask]

        d_non_lineaire = d_non_lineaire[mask]
        v_non_lineaire = v_non_lineaire[mask]
        a_non_lineaire = a_non_lineaire[mask]
        
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
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.plot(t, v_non_lineaire, color="#009CA6")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity")
            ax.set_title(f"Velocity time history - Non Linear model - {selected_component}")
            ax.grid()
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots()
            ax.plot(t, a_non_lineaire, color="#1C2D3F")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration")
            ax.set_title(f"Acceleration time history - Non Linear model - {selected_component}") 
            ax.grid()
            st.pyplot(fig)
        
        
    elif graphique_choisi == "SDOF Structural response - Non Linear Model with Friction":    
        
        a_non_lineaire_friction, v_non_lineaire_friction, d_non_lineaire_friction = newmark_non_lineaire_friction()
            
        # Sauvegarde des résultats
        st.session_state.results = {"t": t, "F": F, "d_non_lineaire_friction": d_non_lineaire_friction, "v_non_lineaire_friction": v_non_lineaire_friction, "a_non_lineaire_friction": a_non_lineaire_friction}
        st.session_state.last_params = params_key
        
        # Récupération des résultats depuis session_state
        t = st.session_state.results["t"]
        F = st.session_state.results["F"]

        d_non_lineaire_friction = st.session_state.results["d_non_lineaire_friction"]
        v_non_lineaire_friction = st.session_state.results["v_non_lineaire_friction"]
        a_non_lineaire_friction = st.session_state.results["a_non_lineaire_friction"]
        
        # Indices correspondant à la plage de temps sélectionnée
        mask = (t >= selected_range[0]) & (t <= selected_range[1])

        # Filtrage des données
        t = t[mask]
        F = F[mask]

        d_non_lineaire_friction = d_non_lineaire_friction[mask]
        v_non_lineaire_friction = v_non_lineaire_friction[mask]
        a_non_lineaire_friction = a_non_lineaire_friction[mask]
        
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
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            ax.plot(t, v_non_lineaire_friction, color="#009CA6")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity")
            ax.set_title(f"Velocity time history - Non Linear model - Friction - {selected_component}")
            ax.grid()
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots()
            ax.plot(t, a_non_lineaire_friction, color="#1C2D3F")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration")
            ax.set_title(f"Acceleration time history - Non Linear model - Friction - {selected_component}") 
            ax.grid()
            st.pyplot(fig)
        
    elif graphique_choisi == "Acceleration and Displacement spectrum":   
        # Calcul du spectre de Fourrier
        T0_list = np.linspace(0.02, 20, 50)
        
        Sa, Sv, Sd = newmark_spectrum()
            
        # Sauvegarde des résultats
        st.session_state.results = {"t": t, "F": F, "Sd": Sd, "Sv": Sv, "Sa": Sa, "T0_list": T0_list}
        st.session_state.last_params = params_key
        
        # Récupération des résultats depuis session_state
        t = st.session_state.results["t"]
        F = st.session_state.results["F"]

        T0_list = st.session_state.results["T0_list"]

        Sd = st.session_state.results["Sd"]
        Sv = st.session_state.results["Sv"]
        Sa = st.session_state.results["Sa"]
        
        # Indices correspondant à la plage de temps sélectionnée
        mask = (t >= selected_range[0]) & (t <= selected_range[1])

        # Filtrage des données
        t = t[mask]
        F = F[mask]
        
        # Enregistrement du séisme et spectre de réponse
        st.markdown("Acceleration and Displacement spectrum")
        
        
        col1, col2 = st.columns(2)
            
        with col1:
            fig, ax = plt.subplots()
            ax.plot(T0_list, Sa, color="#002B45")
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("Peak Acceleration")
            ax.set_title(f"Acceleration response spectrum - {selected_component}")
            ax.set_xscale(scale)
            ax.grid()
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots()
            ax.plot(T0_list, Sd, color="#002B45")
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("Peak Displacement")
            ax.set_title(f"Displacement response spectrum  - {selected_component}")
            ax.set_xscale(scale)
            ax.grid()
            st.pyplot(fig) 


    elif graphique_choisi == "Earthquake input":
        
        st.markdown("Earthquake input")
        
        fig, ax = plt.subplots()
        ax.plot(t, F, color="#0072CE")
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Force(N)")
        ax.set_title(f"Ground acceleration - {selected_component}")
        ax.grid()
        st.pyplot(fig)


    elif graphique_choisi == "Te Puni building floor reaction":
        # Calcul du spectre de Fourrier des étages
        T0_list_etage = np.linspace(0.02, 1, 50)
        
        Sa_etage = newmark_spectrum_etage()
        
        # Sauvegarde des résultats
        st.session_state.results = {"t": t, "F": F, "T0_list_etage": T0_list_etage, "Sa_etage": Sa_etage}
        st.session_state.last_params = params_key
        
        # Récupération des résultats depuis session_state
        t = st.session_state.results["t"]
        F = st.session_state.results["F"]

        T0_list_etage = st.session_state.results["T0_list_etage"]

        Sa_etage = st.session_state.results["Sa_etage"]

        # Indices correspondant à la plage de temps sélectionnée
        mask = (t >= selected_range[0]) & (t <= selected_range[1])
        
        # Filtrage des données
        t = t[mask]
        F = F[mask]
        
        col1, col2 = st.columns(2) 
        
        # Affichage du graphiques pour les étages
        with col1:
            st.markdown("Te Puni building floor reaction")

            fig, ax = plt.subplots()
            for j in range(nb_etage):
                etage = nb_etage - j
                ax.plot(T0_list_etage, Sa_etage[j], label=f"Floor {j+1}")
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("Peak Acceleration")
            ax.set_title("Acceleration response spectrum per floor")
            ax.set_xscale(scale)
            ax.grid()
            ax.legend()
            st.pyplot(fig)
          

        # Hauteur de chaque étage
        Hauteur_Te_Puni = np.linspace(0, 3*(nb_etage-1), nb_etage)  # Exemple : 3 m par étage

        # Calcul du max du spectre par étage
        acc_data_etage = []
        for l in range(1, n_floors+1):
            acc_data_etage.append(pd.to_numeric(df_etage.iloc[:, l], errors='coerce').values)
            
        max_peak_acceleration = [max(Ac) for Ac in acc_data_etage]
        Amplitude_factor = np.abs(max_peak_acceleration / max(acc_data))

        # Affichage
        with col2:
            st.markdown("Maximum acceleration amplitude factor per floor")
            fig, ax = plt.subplots()
            ax.plot(Amplitude_factor, Hauteur_Te_Puni, marker="o", color="#1C2D3F")
            ax.set_xlabel("Acceleration amplitude factor (m/s²)")
            ax.set_ylabel("Height (m)")
            ax.set_title("Te Puni building - Max spectral acceleration per floor")
            ax.grid(True)
            st.pyplot(fig)
        
   
    elif graphique_choisi == "Stiffness":    
        def raideur_non_lineaire(d_non_lineaire):
            a_non_lineaire, v_non_lineaire, d_non_lineaire = newmark_non_lineaire()
            F_raideur_non_lineaire = np.zeros(n)
            for i in range(n-1):
                F_raideur_non_lineaire[i] = K * d_non_lineaire[i] + K3 * d_non_lineaire[i] ** 3
            return F_raideur_non_lineaire

        F_raideur_non_lineaire = raideur_non_lineaire(d_non_lineaire)

        st.markdown("Stiffness")   

        fig, ax = plt.subplots()
        ax.plot(d_non_lineaire, F_raideur_non_lineaire, color="#002B45")
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Stiffness force")
        ax.set_title(f"Stiffness force - Non Linear model - {selected_component}")
        ax.grid()
        st.pyplot(fig)