import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
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

# Upload du fichier CSV ou Excel
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                # Lecture auto avec détection de séparateur
                df = pd.read_csv(uploaded_file, skiprows=8, engine="python", sep=None)
            except Exception as e:
                st.error(f"Échec de lecture CSV : {e}")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file, skiprows=8)

        #st.write("Colonnes détectées après lecture :", df.columns.tolist())


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
            st.error("Aucune composante verticale ou horizontale détectée dans le fichier.")
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
            st.error("Composante sélectionnée introuvable.")
            st.stop()
 
        if time_col is None or acc_col is None:
            st.error("Colonnes 'Time' et 'Vertical' non détectées.")
            st.stop()

        # Conversion du temps depuis format ISO 8601 en secondes écoulées
        if df[time_col].dtype == object:
            try:
                # Convertir en datetime
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

                # Vérifier qu'aucune conversion n'a échoué
                if df[time_col].isnull().any():
                    st.error("Conversion datetime impossible pour certains temps (valeurs invalides).")
                    st.stop()

                # Calculer les secondes écoulées depuis le premier timestamp
                time_zero = df[time_col].iloc[0]
                df[time_col] = (df[time_col] - time_zero).dt.total_seconds()

            except Exception as e:
                st.error(f"Conversion de 'Time' impossible : {e}")
                st.stop()

        # Extraction des valeurs numériques
        time_data = pd.to_numeric(df[time_col], errors='coerce').values
        acc_data = pd.to_numeric(df[acc_col], errors='coerce').values

        # Filtrer les NaN
        valid = ~np.isnan(time_data) & ~np.isnan(acc_data)
        time_data = time_data[valid]
        acc_data = acc_data[valid]

        #st.success(f"Données valides détectées : {len(time_data)} points")
        #st.write("Exemple de données :", pd.DataFrame({"Time (s)": time_data[:5],"Acceleration": acc_data[:5]}))

    except Exception as e:
        st.error(f"Erreur : {e}")
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

params_key = (M, K, zeta, T, selected_component)

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

# Définition de la fréquence propre du système
W = (K * M) ** (1 / 2)
st.sidebar.markdown(f"ω₀ : Natural frequency : **{W:.2f} rad/s**")
if W == 0:
    st.error("Error: ω₀ is zero (check M and K)")
    st.stop()

# Définition du pas de temps
dt1 = 2 / W * (1 / (1 - 2 * beta)) ** (1 / 2)
dt = dt1 * 0.9  # 90% de la limite de stabilité, par exemple
st.sidebar.markdown(f"dt : Step time the simulation use : **{dt:.5f} s**")

# Gestion des variables temporels
t = np.arange(0, T + dt,
              dt)  # fonction NumPy qui crée un tableau (array) des valeurs du temps espacées régulièrement et on fait + dt pour avoir la durée finale réelle
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

selected_range = st.sidebar.slider("Select the time range to display (s)", time_min, time_max, key="time_range_slider",
                                   step=1.0)

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
    F = - M * accel

    # Initialisation des réponses
    d = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # Conditions initiales
    d[0] = 0
    v[0] = 0
    a[0] = (F[0] - C * v[0] - K * d[0]) / M

    # Pré-calculs
    B = M + K * beta * dt ** 2 + C * gamma * dt
    if B == 0:
        st.error("Error: Denominator B is zero. Try adjusting M, K, damping zeta, or time step dt.")
        st.stop()

    # Newmark
    for i in range(n - 1):
        P = v[i] + ((1 - gamma) * dt) * a[i]
        H = d[i] + dt * v[i] + (1 / 2 - beta) * dt ** 2 * a[i]
        a[i + 1] = (F[i + 1] - K * H - C * P) / B
        v[i + 1] = P + gamma * dt * a[i + 1]
        d[i + 1] = H + beta * dt ** 2 * a[i + 1]

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

# Affichage

# 🔹 Affichage d'un titre si l'utilisateur n'a pas encore uploadé de fichier
if uploaded_file is None:
    selected_component = "Vertical"
    st.markdown("### Example of simulation with default data")
    st.info(
        "You are currently viewing a simulation example with predefined data. To use your own seismic data, import a CSV or Excel file at the top of the page.")

# Première ligne : Force et déplacement
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, F, label="Force (N)", color="#0072CE")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Force(N)")
    ax.set_title(f"Earthquake Modelisation - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(t, d, label="Movement (m)", color="#002B45")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Movement")
    ax.set_title(f"Movement - Newmark Method - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

# Deuxième ligne : vitesse et accélération
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    ax.plot(t, v, label="Velocity (m/s)", color="#009CA6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity - Newmark Method - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    ax.plot(t, a, label="Acceleration (m/s^2)", color="#1C2D3F")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Acceleration - Newmark Method - {selected_component}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

output_df = pd.DataFrame(
    {"Time (s)": t, "Displacement (m)": d, "Velocity (m/s)": v, "Acceleration (m/s²)": a, "Force (N)": F})
csv = output_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download results as CSV", data=csv, file_name='newmark_results.csv', mime='text/csv')

