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
F1 = st.sidebar.number_input("F1 : Amplitude coefficient (N)", min_value=0.0, max_value=100.0, value=1.0, step=1.0, format="%.4f")

d0 = st.sidebar.slider("d : Initial movement (m)", 0.0, 0.5, 0.0, step=0.01)
v0 = st.sidebar.slider("v : Initial velocity (m/s)", 0.0, 1.0, 0.0, step=0.01)

scale = st.sidebar.radio("Period axis scale for response spectra", ("linear", "log"))

friction_enabled = st.sidebar.checkbox("Include Coulomb friction")

mu = st.sidebar.slider("μ : Friction coefficient", 0.0, 2.0, 0.5, step=0.01)

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

    
params_key = (M, K, zeta, T, selected_component, d0, v0, dt, F1, scale, mu)


# Définition du coefficent d'amortissement
C = (2 * (K * M) ** (1 / 2) * zeta) / 100
st.sidebar.markdown(f"C : Damping coefficient : **{C:.2f} Ns/m**")

# Filtrer les NaN
valid_indices = ~np.isnan(time_data) & ~np.isnan(acc_data)  # np.isnan(time_data) Cette fonction renvoie un tableau de booléens (True/False) de même taille que time_data.
# Elle contient True là où les valeurs sont NaN (Not a Number), c’est-à-dire des valeurs manquantes ou invalides.
# L’opérateur ~ est un NON logique (négation). Donc cette expression renvoie True pour les indices valides (non NaN) de time_data.
time_data = time_data[valid_indices]
acc_data = acc_data[valid_indices]

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
    st.error(f"⚠️ Stability condition not met: dt = {dt:.4f} > dtₘₐₓ ≈ {dt1:.4f} s.\nTry reducing `dt` below this limit.")
    st.stop()


# Gestion des variables temporels
t = np.arange(0, T + dt,dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
n = len(t)

# 🔧 MODIF : bornes du slider
time_min = 0.0
time_max = T

# 🔧 MODIF : initialisation de session_state pour le slider et T précédent
if "time_range_slider" not in st.session_state or "previous_T" not in st.session_state:
    st.session_state["time_range_slider"] = (time_min, time_max)
    st.session_state["previous_T"] = T

# 🔧 MODIF : mise à jour du slider si T a changé
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

    # Conditions initiales
    d[0] = d0
    v[0] = v0
    a[0] = (F[0] - C * v[0] - K * d[0]) / M
    
    d_friction[0] = d0
    v_friction[0] = v0
    a_friction[0] = (F[0] - C * v[0] - K * d[0]) / M

    # Pré-calculs
    B = M + K * beta * dt ** 2 + C * gamma * dt
    if B == 0:
        st.error("Error: Denominator B is zero. Try adjusting M, K, damping zeta, or time step dt.")
        st.stop()

    # Newmark
    for i in range(n - 1):
        P = v[i] + ((1 - gamma) * dt) * a[i]
        H = d[i] + dt * v[i] + (1 / 2 - beta) * dt ** 2 * a[i]
        
        # Friction régulière (approximation continue)
        friction = mu * N_force * np.tanh(v[i] / v_eps) if friction_enabled else 0.0

        # Force totale (avec frottement)
        F_friction[i+1] = F[i+1] - friction

        a[i + 1] = (F[i + 1] - K * H - C * P) / B 
        v[i + 1] = P + gamma * dt * a[i + 1] 
        d[i + 1] = H + beta * dt ** 2 * a[i + 1] 
        
        a_friction[i + 1] = (F_friction[i + 1] - K * H - C * P) / B 
        v_friction[i + 1] = P + gamma * dt * a[i + 1] 
        d_friction[i + 1] = H + beta * dt ** 2 * a[i + 1] 
        
        
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
    

    # Sauvegarde des résultats
    st.session_state.results = {"t": t, "F": F, "d": d, "v": v, "a": a, "Sd": Sd, "Sv": Sv, "Sa": Sa, "T0_list": T0_list, "a_friction": a_friction}
    st.session_state.last_params = params_key

# Récupération des résultats depuis session_state
t = st.session_state.results["t"]
F = st.session_state.results["F"]
d = st.session_state.results["d"]
v = st.session_state.results["v"]
a = st.session_state.results["a"]

a_friction = st.session_state.results["a_friction"]

T0_list = st.session_state.results["T0_list"]

Sd = st.session_state.results["Sd"]
Sv = st.session_state.results["Sv"]
Sa = st.session_state.results["Sa"]

# Indices correspondant à la plage de temps sélectionnée
mask = (t >= selected_range[0]) & (t <= selected_range[1])

# Filtrage des données
t = t[mask]
F = F[mask]
d = d[mask] 
v = v[mask]
a = a[mask] 

a_friction = a_friction[mask]



# Affichage

# 🔹 Affichage d'un titre si l'utilisateur n'a pas encore uploadé de fichier
if uploaded_file is None:
    selected_component = "Vertical"
    st.markdown("### Example of simulation with default data")
    st.info(
        "You are currently viewing a simulation example with predefined data. To use your own seismic data, import a CSV or Excel file at the top of the page.")


# Première ligne : Force et déplacement

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
    #ax.set_xscale("log")
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
    #ax.set_xscale("log")
    ax.grid()
    ax.legend()
    st.pyplot(fig)    
    
st.markdown("SDOF Structural response")   

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, d, color="#002B45")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Displacement time history - Newmark Method - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(t, v, color="#009CA6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity time history - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    ax.plot(t, a, color="#1C2D3F")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Acceleration time history - {selected_component}") 
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    
if friction_enabled:
    friction_values = mu * N_force * np.tanh(v / v_eps)
    fig, ax = plt.subplots()
    ax.plot(t, a_friction, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Friction force (N)")
    ax.set_title("Friction force over time")
    ax.grid()
    st.pyplot(fig)
     
      
output_df = pd.DataFrame(
    {"Time (s)": t, "Displacement (m)": d, "Velocity (m/s)": v, "Acceleration (m/s²)": a, "Force (N)": F})
csv = output_df.to_csv(index=False).encode('utf-8')
st.download_button("Download results as CSV", data=csv, file_name='newmark_results.csv', mime='text/csv')
