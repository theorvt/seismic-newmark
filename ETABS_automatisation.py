import comtypes.client
import pandas as pd
import os

# Paramètres

# Chemin du modèle ETABS
model_path = r"C:\Users\Theo\Documents\Modele_ETABS.edb"

# Dossier contenant les fichiers séismes (accélérations X, Y)
seism_folder = r"C:/Users/theorivet/Documents/GitHub/seismic-newmark/donnee_caroline_1_seisme"

# Liste des fichiers séisme (ex: 570)
#permet de lister le contenu d'un répertoire spécifié en renvoyant une liste des noms de fichiers et de sous-répertoires contenus dans ce répertoire
seism_files = [f for f in os.listdir(seism_folder) if f.endswith(".txt")]

# Nom de l’étage que tu veux analyser (par ex. étage 9)
target_floor = "Floor9"

# Résumé global
results_summary = []

# OUVRIR ETABS

ETABSObject = comtypes.client.CreateObject("CSI.ETABS.API.ETABSObject")
ETABSObject.ApplicationStart()

SapModel = ETABSObject.SapModel
SapModel.InitializeNewModel()

# Ouvrir modèle
ret = SapModel.File.OpenFile(model_path)


# BOUCLE SUR LES SÉISMES

for i, seism_file in enumerate(seism_files, 1): # enumerate permet de parcourir une liste en donnant à la fois l’index et l’élément.
    eq_name = f"EQ_{i}"  # Nom du cas de charge
    eq_path = os.path.join(seism_folder, seism_file)  # eq_path contiendra le chemin complet vers ton fichier sismique, que tu pourras ensuite ouvrir avec pandas, open, etc.

    print(f"==> Traitement du séisme {eq_name} ({seism_file})")

    # 1. Créer un cas Time History (linéaire)
    ret = SapModel.LoadCases.ModHistLinear.SetCase(eq_name)
    # SapModel C’est l’objet principal de ton modèle ETABS (il contient tout : géométrie, matériaux, cas de charges, combinaisons, etc.).
    # LoadCases C’est la partie de l’API qui gère les cas de charges dans ETABS.
    # ModHistLinear Cela correspond aux cas d’analyse dynamique de type "Linear Modal History" (= Historique modal linéaire). En clair : tu définis un cas où la réponse du bâtiment est calculée par superposition modale à partir d’un signal sismique (accélération vs temps).
    # SetCase(eq_name) Cette méthode crée un nouveau cas de charge dynamique (historique modal linéaire) et lui donne le nom eq_name. eq_name est une chaîne de caractères → le nom que tu donnes à ce cas (ex. "ElCentro"). SetCase initialise ce cas dans ETABS. Tu devras ensuite configurer les détails (fichier de séisme, échelle, amortissement, etc.). 5. ret Dans l’API ETABS, la plupart des fonctions renvoient un code retour : 0 → succès ≠0 → erreur  Donc ici ret permet de vérifier si la création du cas s’est bien passée.
    
    # Ajouter accélérations (X ou Y selon fichier)
    # Ici il faut préciser comment sont formatés tes séismes
    # Exemple: charger directement le fichier dans ETABS
    ret = SapModel.Func.FuncTH.SetFromFile(eq_name, eq_path)
    # Crée une fonction de type Time History appelée eq_name, en lisant les données contenues dans le fichier eq_path.

    # 2. Lancer l’analyse
    SapModel.Analyze.RunAnalysis()

    # 3. Extraire résultats

    # Accélération
    ret, obj, elm, cases, step_type, step_num, acc = SapModel.Results.Acceleration(target_floor, eq_name)

    # Déplacement
    ret, obj, elm, cases, step_type, step_num, disp = SapModel.Results.Displ(target_floor, eq_name)

    # Vitesse
    ret, obj, elm, cases, step_type, step_num, vel = SapModel.Results.Velocity(target_floor, eq_name)

    # 4. Post-traitement
    max_acc = max(abs(a) for a in acc)
    max_vel = max(abs(v) for v in vel)
    max_disp = max(abs(d) for d in disp)

    # Exemple : PGA (acc max au sol)
    PGA = max_acc / 9.81  # en g

    # Spectre de réponse : tu peux soit le calculer en Python (Sd, Sv, Sa)
    #    soit utiliser ETABS pour l’exporter directement si disponible.
    #    Ici je laisse un placeholder
    PSA_05s = None
    PSA_1s = None

    # results_summary.append(...) results_summary est probablement une liste vide au départ → results_summary = []. .append(...) sert à ajouter un nouvel élément à cette liste.
    results_summary.append({"ID": seism_file,"PGA(g)": PGA,"Max Acc (m/s²)": max_acc,"Max Vel (m/s)": max_vel,
                            "Max Disp (m)": max_disp,"PSA(0.5s)": PSA_05s,"PSA(1s)": PSA_1s})

# SAUVEGARDE DES RÉSULTATS

df_results = pd.DataFrame(results_summary)
df_results.to_excel("etabs_summary_results.xlsx", index=False)
print("Analyse terminée, résultats sauvegardés dans etabs_summary_results.csv")