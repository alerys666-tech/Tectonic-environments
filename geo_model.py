from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from matplotlib.patches import Ellipse

plt.rcParams["figure.dpi"] = 140

LABEL_COL = "Ambiente_Tectonico"
THRESHOLD_TRANSICIONAL = 0.55
np.random.seed(42)

# 1) UTILIDADES: RATIOS + LECTURA UNIVERSAL
# ============================================================
def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"Nb","Y"}.issubset(df.columns):   df["Nb_Y"]  = safe_div(df["Nb"], df["Y"])
    if {"Zr","Y"}.issubset(df.columns):   df["Zr_Y"]  = safe_div(df["Zr"], df["Y"])
    if {"Th","Yb"}.issubset(df.columns):  df["Th_Yb"] = safe_div(df["Th"], df["Yb"])
    if {"Nb","Yb"}.issubset(df.columns):  df["Nb_Yb"] = safe_div(df["Nb"], df["Yb"])
    if {"La","Yb"}.issubset(df.columns):  df["La_Yb"] = safe_div(df["La"], df["Yb"])
    if {"Sm","Yb"}.issubset(df.columns):  df["Sm_Yb"] = safe_div(df["Sm"], df["Yb"])
    if {"La","Sm"}.issubset(df.columns):  df["La_Sm"] = safe_div(df["La"], df["Sm"])
    if {"Th","Nb"}.issubset(df.columns):  df["Th_Nb"] = safe_div(df["Th"], df["Nb"])
    if {"Na2O","K2O"}.issubset(df.columns):
        df["Na2OplusK2O"] = df["Na2O"] + df["K2O"]
    return df

def read_any(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_excel(path)

    rename_map = {
        "SiO2 (wt%)":"SiO2","TiO2 (wt%)":"TiO2","Al2O3 (wt%)":"Al2O3",
        "FeOt (wt%)":"FeOt",
        "MgO (wt%)":"MgO","CaO (wt%)":"CaO","Na2O (wt%)":"Na2O","K2O (wt%)":"K2O","P2O5 (wt%)":"P2O5",
        "MnO (wt%)":"MnO",
        "Rb (ppm)":"Rb","Sr (ppm)":"Sr","Ba (ppm)":"Ba",
        "Nb (ppm)":"Nb","Ta (ppm)":"Ta","Zr (ppm)":"Zr","Hf (ppm)":"Hf","Y (ppm)":"Y",
        "Th (ppm)":"Th","U (ppm)":"U","Pb (ppm)":"Pb",
        "La (ppm)":"La","Ce (ppm)":"Ce","Nd (ppm)":"Nd","Sm (ppm)":"Sm","Eu (ppm)":"Eu",
        "Yb (ppm)":"Yb","Lu (ppm)":"Lu",
        "Fe2O3T (wt%)":"Fe2O3T",
        "FeO (wt%)":"FeO",
        "Fe2O3T":"Fe2O3T",
        "FeO":"FeO",
    }
    df = df.rename(columns=rename_map)

    # FeOt si no existe (desde Fe2O3T o FeO)
    if "FeOt" not in df.columns:
        if "Fe2O3T" in df.columns:
            df["FeOt"] = pd.to_numeric(df["Fe2O3T"], errors="coerce") * 0.8998
        elif "FeO" in df.columns:
            df["FeOt"] = pd.to_numeric(df["FeO"], errors="coerce")
        else:
            df["FeOt"] = np.nan

    num_cols = [
        "SiO2","TiO2","Al2O3","FeOt","MnO","MgO","CaO","Na2O","K2O","P2O5",
        "Rb","Sr","Ba","Nb","Ta","Zr","Hf","Y","Th","U","Pb",
        "La","Ce","Nd","Sm","Eu","Yb","Lu"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_ratios(df)
    return df

# 2) FEATURES DEL MODELO (alta cobertura)
# ============================================================
CORE = [
    "SiO2","TiO2","Al2O3","FeOt","MgO","CaO","Na2O","K2O","P2O5",
    "Rb","Sr","Ba","Nb","Zr","Y","Th",
    "La","Sm","Yb"
]
RATIO_WANTED = [
    "Nb_Y","Zr_Y","Th_Yb","Nb_Yb","La_Yb","Sm_Yb","La_Sm","Na2OplusK2O","Th_Nb"
]

def build_train_cols(df: pd.DataFrame):
    base_cols  = [c for c in CORE if c in df.columns]
    ratio_cols = [r for r in RATIO_WANTED if r in df.columns]
    return base_cols + ratio_cols

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=1200,
            max_depth=22,
            min_samples_leaf=3,
            max_features=0.6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

# 4) PASO A PASO: ENTRENAR -> PREDECIR -> GRAFICAR -> EXPORTAR
# ============================================================

# -------------------------
# (A) SUBIR ENTRENAMIENTO
# -------------------------
print("📌 PASO 1/2: Sube tu archivo de ENTRENAMIENTO (debe tener 'Ambiente_Tectonico').")
upl_train = files.upload()
train_path = Path(list(upl_train.keys())[0])
print("✅ Entrenamiento:", train_path)

df_train = read_any(train_path)
if LABEL_COL not in df_train.columns:
    raise RuntimeError("❌ Tu archivo de entrenamiento NO tiene la columna 'Ambiente_Tectonico'.")

train_cols = build_train_cols(df_train)
print("\n✅ Variables usadas (train_cols):")
print(train_cols)
print("N variables:", len(train_cols))

df_train = df_train.dropna(subset=[LABEL_COL]).copy()
X = df_train[train_cols]
y = df_train[LABEL_COL].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = build_model()
model.fit(X_tr, y_tr)

print("\n=====================")
print("✅ Evaluación del modelo (test) - 5 clases")
print("=====================")
print(classification_report(y_te, model.predict(X_te)))

# ============================================================
# VALIDACIÓN GRÁFICA DEL MODELO
# - Matriz de confusión
# - Accuracy train vs test
# - Confianza: entrenado vs validado
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# -------------------------
# 1) MATRIZ DE CONFUSIÓN (TEST)
# -------------------------
y_test_pred = model.predict(X_te)

cm = confusion_matrix(y_te, y_test_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Matriz de confusión – Datos de validación (Test)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -------------------------
# 2) ACCURACY: TRAIN vs TEST
# -------------------------
acc_train = accuracy_score(y_tr, model.predict(X_tr))
acc_test  = accuracy_score(y_te, model.predict(X_te))

plt.figure(figsize=(5,4))
plt.bar(["Entrenamiento", "Validación"], [acc_train, acc_test])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy del modelo: Entrenado vs Validado")

for i, v in enumerate([acc_train, acc_test]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()

print(f"✔ Accuracy entrenamiento: {acc_train:.3f}")
print(f"✔ Accuracy validación   : {acc_test:.3f}")

# -------------------------
# 3) CONFIANZA: TRAIN vs TEST
# -------------------------
proba_train = model.predict_proba(X_tr).max(axis=1)
proba_test  = model.predict_proba(X_te).max(axis=1)

plt.figure(figsize=(7,4))
plt.hist(proba_train, bins=30, alpha=0.6, label="Entrenamiento")
plt.hist(proba_test,  bins=30, alpha=0.6, label="Validación")
plt.axvline(THRESHOLD_TRANSICIONAL, color="red", linestyle="--",
            label="Umbral Transicional")
plt.xlabel("Confianza (probabilidad máxima)")
plt.ylabel("Frecuencia")
plt.title("Distribución de confianza: Entrenado vs Validado")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# (B) SUBIR ARCHIVO NUEVO (PREDICCIÓN + PROBAS + AMBIGÜEDAD)
# -------------------------
print("\n📌 PASO 2/2: Sube tu archivo NUEVO para predecir (Excel o CSV).")
upl_new = files.upload()
new_path = Path(list(upl_new.keys())[0])
print("✅ Nuevo:", new_path)

# Leer archivo
df_new = read_any(new_path)

# Asegurar columnas del entrenamiento
for c in train_cols:
    if c not in df_new.columns:
        df_new[c] = np.nan

# (Opcional) recalcular ratios si tu función existe (por si read_any no lo hizo)
if "add_ratios" in globals():
    df_new = add_ratios(df_new)

# Features para el modelo (mismas filas)
X_new = df_new[train_cols].copy()

# Predicción + probabilidades
proba = model.predict_proba(X_new)
pred  = model.predict(X_new)
classes = model.classes_

df_new["Pred_Ambiente"] = pred
df_new["Confianza"] = proba.max(axis=1)

# Regla "Transicional" por umbral de confianza (tu lógica original)
df_new["Pred_Final"] = np.where(
    df_new["Confianza"] >= THRESHOLD_TRANSICIONAL,
    df_new["Pred_Ambiente"],
    "Transicional"
)

# Guardar todas las probabilidades por clase
df_proba = pd.DataFrame(
    proba,
    columns=[f"Proba_{c}" for c in classes],
    index=df_new.index
)
df_new = pd.concat([df_new, df_proba], axis=1)

# Top2 + Delta + Ambiguo (ambigüedad por cercanía Top1 vs Top2)
prob_cols = [f"Proba_{c}" for c in classes]
arr = df_new[prob_cols].values
order = np.argsort(arr, axis=1)[:, ::-1]

top1 = order[:, 0]
top2 = order[:, 1]

df_new["Top1_Clase"] = [classes[i] for i in top1]
df_new["Top1_Prob"]  = arr[np.arange(len(df_new)), top1]
df_new["Top2_Clase"] = [classes[i] for i in top2]
df_new["Top2_Prob"]  = arr[np.arange(len(df_new)), top2]
df_new["Delta_12"]   = df_new["Top1_Prob"] - df_new["Top2_Prob"]

DELTA_TH = 0.15
df_new["Ambiguo"] = df_new["Delta_12"] < DELTA_TH

print("\n✅ Predicciones listas.")
print("Total filas:", len(df_new))
print("Distribución Pred_Final:")
print(df_new["Pred_Final"].value_counts())
print("\nAmbiguos:")
print(df_new["Ambiguo"].value_counts())

df_amb = df_new[df_new["Ambiguo"]].copy()
df_noamb = df_new[~df_new["Ambiguo"]].copy()

print("Ambiguos:", len(df_amb))
print("No ambiguos:", len(df_noamb))

conf_amb = (
    df_amb
    .groupby(["Top1_Clase", "Top2_Clase"])
    .size()
    .sort_values(ascending=False)
)

conf_amb.head(20)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11,5), sharey=True)

ax[0].scatter(df_noamb["Nb_Y"], df_noamb["Zr_Y"], alpha=0.3)
ax[0].set_title("No ambiguos")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Nb/Y"); ax[0].set_ylabel("Zr/Y")
ax[0].grid(True, alpha=0.3)

ax[1].scatter(df_amb["Nb_Y"], df_amb["Zr_Y"], alpha=0.6, color="crimson")
ax[1].set_title("Ambiguos")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("Nb/Y")
ax[1].grid(True, alpha=0.3)

plt.suptitle("Nb/Y vs Zr/Y — Ambiguos vs No ambiguos")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.hist(df_amb["Top1_Prob"], bins=30, alpha=0.7, label="Top1")
plt.hist(df_amb["Top2_Prob"], bins=30, alpha=0.7, label="Top2")
plt.xlabel("Probabilidad")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades en ambiguos")
plt.legend()
plt.tight_layout()
plt.show()

vars_diag = [
    "SiO2","MgO","FeOt","TiO2",
    "Nb_Y","Zr_Y","Th_Nb","La_Yb"
]
vars_diag = [v for v in vars_diag if v in df_new.columns]

summary_amb = pd.DataFrame({
    "Media_Ambiguos": df_amb[vars_diag].mean(),
    "Media_NoAmbiguos": df_noamb[vars_diag].mean(),
    "Ratio_Amb/NoAmb": df_amb[vars_diag].mean() / df_noamb[vars_diag].mean()
})

summary_amb

import numpy as np
import matplotlib.pyplot as plt

def iter_groups(df, group_col):
    dd = df.copy()
    if (group_col is None) or (group_col not in dd.columns):
        return []
    # ordena y deja "Transicional" al final
    groups = [str(g) for g in dd[group_col].dropna().unique()]
    non_t = [g for g in groups if g.lower() != "transicional"]
    t = [g for g in groups if g.lower() == "transicional"]
    return sorted(non_t) + t
def pick_group_col(df):
    for c in [
        "Pred_Final",
        "Pred_Ambiente",
        "Ambiente_Tectonico",
        "Ambiente",
        "Grupo",
        "Sample SubType",
        "Sample Type"
    ]:
        if c in df.columns:
            return c
    return None

def harker_suite_facets(df, group_col=None, ncols=3, s=10, alpha=0.35,
                       ellipses=False, trend=False, max_points=1200, seed=42):
    if "SiO2" not in df.columns:
        print("⚠️ Harker: falta SiO2.")
        return

    oxides = [c for c in ["TiO2","Al2O3","FeOt","MgO","CaO","Na2O","K2O","P2O5"] if c in df.columns]
    if not oxides:
        print("⚠️ Harker: faltan óxidos.")
        return

    if (group_col is None) or (group_col not in df.columns):
        print("⚠️ No hay columna de grupos para facetear.")
        return

    groups = iter_groups(df, group_col)
    if not groups:
        print("⚠️ No hay grupos.")
        return

    # por cada óxido, crea una figura (grid) con ambientes
    rng = np.random.default_rng(seed)
    for ox in oxides:
        n = len(groups)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.0*nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for i, gname in enumerate(groups):
            ax = axes[i]
            g = df[df[group_col].astype(str) == str(gname)].dropna(subset=["SiO2", ox]).copy()

            if len(g) > max_points:
                g = g.sample(max_points, random_state=seed)

            ax.scatter(g["SiO2"], g[ox], s=s, alpha=alpha, rasterized=True)

            # opcional: elipse/tendencia (recomendado: apagado por defecto)
            if ellipses and len(g) >= 20:
                add_cov_ellipse(ax, g["SiO2"].values, g[ox].values, n_std=1.5, alpha=0.10, lw=1.0)
            if trend and len(g) >= 8:
                add_trend_line(ax, g["SiO2"].values, g[ox].values)

            ax.set_title(str(gname), fontsize=9)
            ax.grid(True, alpha=0.2)

        # apagar ejes vacíos
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Harker por ambiente: SiO2 vs {ox}", y=1.02, fontsize=12)
        fig.supxlabel("SiO2 (wt%)")
        fig.supylabel(f"{ox} (wt%)")
        plt.tight_layout()
        plt.show()

group_col = pick_group_col(df_new)
harker_suite_facets(df_new, group_col=group_col, ncols=3, ellipses=False, trend=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Clasificación simple TAS (aprox) SOLO para etiqueta de tipo de roca
def tas_rock_type_simple(sio2, alk):
    """
    Clasificación aproximada por SiO2 y álcalis totales (Na2O+K2O).
    No reemplaza polígonos TAS oficiales; es para visualizar y rotular.
    """
    if pd.isna(sio2) or pd.isna(alk):
        return np.nan

    # rangos comunes (aprox)
    if sio2 < 45:
        return "Basalto (bajo SiO2)"
    elif sio2 < 52:
        return "Basalto"
    elif sio2 < 57:
        return "Andesita basáltica"
    elif sio2 < 63:
        return "Andesita"
    elif sio2 < 69:
        return "Dacita"
    else:
        return "Riolita"

# 2) TAS base (tu versión)
def plot_tas_fields_clean(ax):
    ax.set_xlim(35, 80)
    ax.set_ylim(0, 16)

    for xv in [41, 45, 52, 57, 63, 69]:
        ax.axvline(xv, linewidth=0.8, alpha=0.25)

    xs = np.array([41, 45, 52, 57, 63, 69, 77])
    ys = np.array([3.0, 5.0, 5.5, 6.0, 7.0, 8.0, 1.0])
    ax.plot(xs, ys, linewidth=1.0, alpha=0.6)

    ax.set_xlabel("SiO2 (wt%)")
    ax.set_ylabel("Na2O + K2O (wt%)")

# 3) Orden de ambientes y helper
def iter_groups(df, group_col):
    groups = [str(g) for g in df[group_col].dropna().unique()]
    non_t = [g for g in groups if g.lower() != "transicional"]
    t = [g for g in groups if g.lower() == "transicional"]
    return sorted(non_t) + t

# 4) TAS facets: panel por ambiente + color por tipo de roca
def tas_facets_by_rocktype(df, group_col, ncols=3, s=10, alpha=0.35,
                           max_points=2000, seed=42, show_legend=True):
    need = {"SiO2", "Na2O", "K2O"}
    if not need.issubset(df.columns):
        print("⚠️ TAS: faltan SiO2, Na2O o K2O.")
        return
    if (group_col is None) or (group_col not in df.columns):
        print("⚠️ No hay columna de ambiente (group_col).")
        return

    dd = df.dropna(subset=["SiO2","Na2O","K2O", group_col]).copy()
    if dd.empty:
        print("⚠️ TAS: sin datos.")
        return

    dd["Na2OplusK2O"] = dd["Na2O"] + dd["K2O"]
    dd["Tipo_Roca_TAS"] = dd.apply(lambda r: tas_rock_type_simple(r["SiO2"], r["Na2OplusK2O"]), axis=1)

    groups = iter_groups(dd, group_col)
    n = len(groups)
    nrows = int(np.ceil(n / ncols))

    # categorías de roca para un orden estable
    rock_order = ["Basalto (bajo SiO2)","Basalto","Andesita basáltica","Andesita","Dacita","Riolita"]
    dd["Tipo_Roca_TAS"] = pd.Categorical(dd["Tipo_Roca_TAS"], categories=rock_order, ordered=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.3*nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    # Para leyenda única (global)
    handles_labels = {}

    for i, gname in enumerate(groups):
        ax = axes[i]
        g = dd[dd[group_col].astype(str) == str(gname)].copy()

        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        plot_tas_fields_clean(ax)

        # pinta por tipo de roca
        for rt, gg in g.groupby("Tipo_Roca_TAS"):
            if pd.isna(rt) or gg.empty:
                continue
            sc = ax.scatter(gg["SiO2"], gg["Na2OplusK2O"],
                            s=s, alpha=alpha, rasterized=True, label=str(rt))
            handles_labels[str(rt)] = sc

        ax.set_title(str(gname), fontsize=9)
        ax.grid(True, alpha=0.15)

    # apagar paneles vacíos
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("TAS por ambiente (paneles) + tipo de roca (colores)", y=1.02, fontsize=12)

    # leyenda global (afuera)
    if show_legend and handles_labels:
        fig.legend(handles_labels.values(), handles_labels.keys(),
                   loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=True, fontsize=8, title="Tipo de roca (TAS)")

    plt.tight_layout()
    plt.show()

group_col = pick_group_col(df_new)  # tu columna de ambiente
tas_facets_by_rocktype(df_new, group_col=group_col, ncols=3)

def afm_facets(df, group_col=None, ncols=3, s=10, alpha=0.35, max_points=1500, seed=42):
    need = {"Na2O","K2O","FeOt","MgO"}
    if not need.issubset(df.columns):
        print("⚠️ AFM: faltan Na2O, K2O, FeOt o MgO.")
        return
    if (group_col is None) or (group_col not in df.columns):
        print("⚠️ No hay columna de grupos.")
        return

    dd = df.copy()
    dd["A"] = dd["Na2O"] + dd["K2O"]
    dd["F"] = dd["FeOt"]
    dd["M"] = dd["MgO"]
    dd = dd.dropna(subset=["A","F","M", group_col])
    dd = dd[(dd["A"]>0) & (dd["F"]>0) & (dd["M"]>0)].copy()
    if dd.empty:
        print("⚠️ AFM: sin datos.")
        return

    ssum = dd[["A","F","M"]].sum(axis=1).replace(0, np.nan)
    dd["a"] = dd["A"]/ssum
    dd["f"] = dd["F"]/ssum
    dd["m"] = dd["M"]/ssum
    dd = dd.dropna(subset=["a","f","m"])
    if dd.empty:
        print("⚠️ AFM: sin datos tras normalizar.")
        return

    SQ3_2 = np.sqrt(3)/2
    dd["x"] = dd["m"] + 0.5*dd["f"]
    dd["y"] = dd["f"] * SQ3_2

    groups = iter_groups(dd, group_col)
    n = len(groups)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.4*nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, gname in enumerate(groups):
        ax = axes[i]
        g = dd[dd[group_col].astype(str) == str(gname)].copy()
        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        ax.plot([0,1,0.5,0],[0,0,SQ3_2,0], linewidth=1, alpha=0.8)
        ax.scatter(g["x"], g["y"], s=s, alpha=alpha, rasterized=True)
        ax.set_title(str(gname), fontsize=9)
        ax.axis("off")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("AFM por ambiente (paneles)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

group_col = pick_group_col(df_new)
afm_facets(df_new, group_col=group_col, ncols=3)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# SPIDER PLOTS (BLOQUE ÚNICO)
# ==========================

def pick_group_col(df: pd.DataFrame):
    for c in ["Pred_Final","Pred_Ambiente","Ambiente_Tectonico","Ambiente","Grupo",
              "Sample SubType","Sample Type"]:
        if c in df.columns:
            return c
    return None

def iter_groups(df: pd.DataFrame, group_col: str):
    if (group_col is None) or (group_col not in df.columns):
        return []
    groups = [str(g) for g in df[group_col].dropna().unique()]
    non_t = [g for g in groups if g.lower() != "transicional"]
    t = [g for g in groups if g.lower() == "transicional"]
    return sorted(non_t) + t

def spider_prepare(df: pd.DataFrame, elems):
    # convierte a numérico y limpia <=0 (no válido para log)
    dd = df.copy()
    for e in elems:
        if e in dd.columns:
            dd[e] = pd.to_numeric(dd[e], errors="coerce")
    dd[elems] = dd[elems].mask(dd[elems] <= 0, np.nan)
    dd = dd.dropna(how="all", subset=elems)
    return dd

def spider_all_classes(
    df: pd.DataFrame,
    group_col: str = None,
    elems=None,
    statistic="median",            # "median" o "mean"
    show_iqr=True,                 # banda 25-75%
    min_points=8,                  # mínimo de muestras por clase
    figsize=(10,4.5)
):
    if elems is None:
        elems = [e for e in ["Rb","Ba","Th","Nb","La","Ce","Nd","Sm","Eu","Gd","Tb","Dy","Y","Er","Yb","Lu","Sr","Zr","Hf","Ti","P"] if e in df.columns]

    if len(elems) < 6:
        print("⚠️ Spider: faltan elementos suficientes (mínimo 6).")
        return

    dd = spider_prepare(df, elems)
    if dd.empty:
        print("⚠️ Spider: sin datos.")
        return

    x = np.arange(len(elems))
    fig, ax = plt.subplots(figsize=figsize)

    # si no hay grupos, grafica un perfil global
    if (group_col is None) or (group_col not in dd.columns):
        prof = dd[elems].median(skipna=True) if statistic == "median" else dd[elems].mean(skipna=True)
        ax.plot(x, prof.values, marker="o", linewidth=1.8)

        if show_iqr:
            q1 = dd[elems].quantile(0.25)
            q3 = dd[elems].quantile(0.75)
            ax.fill_between(x, q1.values, q3.values, alpha=0.18)

        ax.set_title("Spider (perfil global)")
    else:
        # todas las clases juntas
        for k, g in dd.groupby(group_col):
            if len(g) < min_points:
                continue
            prof = g[elems].median(skipna=True) if statistic == "median" else g[elems].mean(skipna=True)
            ax.plot(x, prof.values, marker="o", linewidth=1.6, label=str(k))

            if show_iqr:
                q1 = g[elems].quantile(0.25)
                q3 = g[elems].quantile(0.75)
                ax.fill_between(x, q1.values, q3.values, alpha=0.10)

        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02,1), frameon=True)
        ax.set_title("Spider por ambiente (todas las clases)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(elems, rotation=45, ha="right")
    ax.set_ylabel("Concentración (log)")
    ax.grid(True, which="both", alpha=0.20)
    plt.tight_layout()
    plt.show()

def spider_facets(
    df: pd.DataFrame,
    group_col: str,
    elems=None,
    statistic="median",
    show_iqr=True,
    min_points=8,
    ncols=3,
    figsize_per_panel=(3.6, 2.8)
):
    if elems is None:
        elems = [e for e in ["Rb","Ba","Th","Nb","La","Ce","Nd","Sm","Eu","Gd","Tb","Dy","Y","Er","Yb","Lu","Sr","Zr","Hf","Ti","P"] if e in df.columns]

    if len(elems) < 6:
        print("⚠️ Spider: faltan elementos suficientes (mínimo 6).")
        return
    if (group_col is None) or (group_col not in df.columns):
        print("⚠️ Spider facets: no hay columna de grupos.")
        return

    dd = spider_prepare(df, elems)
    dd = dd.dropna(subset=[group_col])
    if dd.empty:
        print("⚠️ Spider facets: sin datos.")
        return

    groups = iter_groups(dd, group_col)
    groups = [g for g in groups if (dd[group_col].astype(str) == str(g)).sum() >= min_points]
    if not groups:
        print("⚠️ Spider facets: ningún grupo cumple min_points.")
        return

    n = len(groups)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0]*ncols, figsize_per_panel[1]*nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(elems))

    for i, gname in enumerate(groups):
        ax = axes[i]
        g = dd[dd[group_col].astype(str) == str(gname)]

        prof = g[elems].median(skipna=True) if statistic == "median" else g[elems].mean(skipna=True)
        ax.plot(x, prof.values, marker="o", linewidth=1.6)

        if show_iqr:
            q1 = g[elems].quantile(0.25)
            q3 = g[elems].quantile(0.75)
            ax.fill_between(x, q1.values, q3.values, alpha=0.18)

        ax.set_title(str(gname), fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.20)

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Spider por ambiente (paneles)", y=1.02, fontsize=12)
    for ax in axes[:min(len(groups), len(axes))]:
        ax.set_xticks(x)
        ax.set_xticklabels(elems, rotation=45, ha="right", fontsize=7)

    fig.supylabel("Concentración (log)")
    plt.tight_layout()
    plt.show()

# ==========================
# USO (CAMBIA df_new si tu df se llama diferente)
# ==========================
group_col = pick_group_col(df_new)
print("🎯 Columna de grupos detectada:", group_col)

# 1) Spider con todas las clases en una sola figura
spider_all_classes(df_new, group_col=group_col, statistic="median", show_iqr=True)

# 2) Spider en paneles (un gráfico por ambiente)
if group_col is not None:
    spider_facets(df_new, group_col=group_col, statistic="median", show_iqr=True, ncols=3)

# (C) GRÁFICOS BÁSICOS (resumen)
# -------------------------
print("\n📊 Distribución Pred_Final (archivo nuevo):")
print(df_new["Pred_Final"].value_counts())

plt.figure(figsize=(7,4))
df_new["Pred_Final"].value_counts().plot(kind="bar")
plt.title("Distribución de ambientes predichos (Pred_Final)")
plt.xlabel("Ambiente")
plt.ylabel("Conteo")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.hist(df_new["Confianza"].dropna(), bins=30)
plt.title("Histograma de confianza del modelo")
plt.xlabel("Confianza (max prob)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

i = 10  # cambia el índice

prob_cols = [c for c in df_new.columns if c.startswith("Proba_")]

print("Probabilidades por ambiente (fila", i, "):")
print(df_new.loc[i, prob_cols])

cols_show = (
    ["Pred_Final", "Top1_Clase", "Top1_Prob", "Top2_Clase", "Top2_Prob"]
    + [c for c in df_new.columns if c.startswith("Proba_")]
)

df_new[cols_show].head(10)

df_amb = df_new[df_new["Ambiguo"]]

df_amb[[
    "Pred_Final",
    "Top1_Clase", "Top1_Prob",
    "Top2_Clase", "Top2_Prob"
] + [c for c in df_new.columns if c.startswith("Proba_")]].head(15)

prob_cols = [c for c in df_new.columns if c.startswith("Proba_")]

df_probs_long = (
    df_new[["Pred_Final", "Ambiguo"] + prob_cols]
    .melt(
        id_vars=["Pred_Final", "Ambiguo"],
        value_vars=prob_cols,
        var_name="Ambiente",
        value_name="Probabilidad"
    )
)

# limpiar nombre
df_probs_long["Ambiente"] = df_probs_long["Ambiente"].str.replace("Proba_", "")

df_probs_long.head()

prob_summary = (
    df_probs_long
    .groupby("Ambiente")["Probabilidad"]
    .agg(["mean", "median", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)

prob_summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prob_cols = [c for c in df_new.columns if c.startswith("Proba_")]
if not prob_cols:
    raise ValueError("No encuentro columnas Proba_*. Asegúrate de haber calculado predict_proba y concatenado df_proba.")

# formato largo (ideal para graficar)
df_probs_long = (
    df_new[["Pred_Final", "Ambiguo"] + prob_cols]
    .melt(
        id_vars=["Pred_Final", "Ambiguo"],
        value_vars=prob_cols,
        var_name="Ambiente",
        value_name="Probabilidad"
    )
)
df_probs_long["Ambiente"] = df_probs_long["Ambiente"].str.replace("Proba_", "", regex=False)

order = (
    df_probs_long.groupby("Ambiente")["Probabilidad"]
    .median()
    .sort_values(ascending=False)
    .index
    .tolist()
)

plt.figure(figsize=(10, 4.8))
plt.boxplot(
    [df_probs_long[df_probs_long["Ambiente"] == a]["Probabilidad"].dropna().values for a in order],
    labels=order,
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Probabilidad")
plt.title("Distribución de probabilidades por ambiente (todas las muestras)")
plt.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

df_probs_amb = df_probs_long[df_probs_long["Ambiguo"] == True].copy()

order_amb = (
    df_probs_amb.groupby("Ambiente")["Probabilidad"]
    .median()
    .sort_values(ascending=False)
    .index
    .tolist()
)

plt.figure(figsize=(10, 4.8))
plt.boxplot(
    [df_probs_amb[df_probs_amb["Ambiente"] == a]["Probabilidad"].dropna().values for a in order_amb],
    labels=order_amb,
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Probabilidad")
plt.title("Distribución de probabilidades por ambiente (solo ambiguos)")
plt.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# tabla Top1 vs Top2 para ambiguos
df_amb = df_new[df_new["Ambiguo"] == True].copy()

ct = pd.crosstab(df_amb["Top1_Clase"], df_amb["Top2_Clase"])

plt.figure(figsize=(7.5, 6.2))
plt.imshow(ct.values, aspect="auto")
plt.xticks(range(ct.shape[1]), ct.columns, rotation=45, ha="right")
plt.yticks(range(ct.shape[0]), ct.index)
plt.colorbar(label="Conteo (ambiguous)")
plt.title("Confusiones en ambiguos: Top1 vs Top2")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.hist(df_new["Delta_12"].dropna(), bins=50)
plt.axvline(0.15, linestyle="--", label="Umbral ambigüedad (0.15)")
plt.xlabel("Delta_12 = Top1_Prob - Top2_Prob")
plt.ylabel("Frecuencia")
plt.title("Distribución de ambigüedad (Delta_12)")
plt.legend()
plt.tight_layout()
plt.show()

import re
import unicodedata

def _norm(s: str) -> str:
    """minúsculas, sin tildes, solo letras/números"""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def detect_lat_lon_cols(df):
    # patrones típicos (LAT)
    lat_patterns = [
        r"^lat$", r"^latitude$", r"^latitud$", r"^y$", r"^north$", r"^northing$",
        r".*lat.*", r".*ycoord.*", r".*coordy.*"
    ]
    # patrones típicos (LON)
    lon_patterns = [
        r"^lon$", r"^lng$", r"^long$", r"^longitude$", r"^longitud$", r"^x$", r"^east$", r"^easting$",
        r".*lon.*", r".*lng.*", r".*long.*", r".*xcoord.*", r".*coordx.*"
    ]

    cols = list(df.columns)
    cols_norm = {c: _norm(c) for c in cols}

    def score_col(cnorm, patterns):
        score = 0
        for p in patterns:
            if re.match(p, cnorm):
                score += 5
            if re.search(p, cnorm):
                score += 2
        return score

    lat_best = max(cols, key=lambda c: score_col(cols_norm[c], lat_patterns), default=None)
    lon_best = max(cols, key=lambda c: score_col(cols_norm[c], lon_patterns), default=None)

    # validar que de verdad haya score > 0
    if score_col(cols_norm.get(lat_best, ""), lat_patterns) == 0:
        lat_best = None
    if score_col(cols_norm.get(lon_best, ""), lon_patterns) == 0:
        lon_best = None

    return lat_best, lon_best
