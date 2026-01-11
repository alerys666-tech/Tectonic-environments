
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# -------------------------
# Config global
# -------------------------
plt.rcParams["figure.dpi"] = 140

LABEL_COL = "Ambiente_Tectonico"
THRESHOLD_TRANSICIONAL = 0.55  # por defecto; en la app puedes sobreescribirlo
DEFAULT_DELTA_TH = 0.15        # umbral ambigüedad Top1-Top2

np.random.seed(42)


# ============================================================
# 1) UTILIDADES: RATIOS + LECTURA UNIVERSAL
# ============================================================
def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Crea ratios comunes si existen las columnas necesarias."""
    df = df.copy()

    if {"Nb", "Y"}.issubset(df.columns):
        df["Nb_Y"] = safe_div(df["Nb"], df["Y"])
    if {"Zr", "Y"}.issubset(df.columns):
        df["Zr_Y"] = safe_div(df["Zr"], df["Y"])
    if {"Th", "Nb"}.issubset(df.columns):
        df["Th_Nb"] = safe_div(df["Th"], df["Nb"])
    if {"La", "Yb"}.issubset(df.columns):
        df["La_Yb"] = safe_div(df["La"], df["Yb"])

    # TAS helper
    if {"Na2O", "K2O"}.issubset(df.columns):
        df["Na2OplusK2O"] = df["Na2O"] + df["K2O"]

    return df


def read_any(path_or_buffer: Union[str, Path, "pd.io.common.FilePathOrBuffer", object]) -> pd.DataFrame:
    """Lee Excel/CSV desde:
    - ruta (str/Path)
    - buffer tipo Streamlit UploadedFile
    """
    if isinstance(path_or_buffer, (str, Path)):
        p = str(path_or_buffer)
        if p.lower().endswith(".csv"):
            df = pd.read_csv(p, low_memory=False)
        else:
            df = pd.read_excel(p)
    else:
        # streamlit UploadedFile o file-like
        name = getattr(path_or_buffer, "name", "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(path_or_buffer, low_memory=False)
        else:
            df = pd.read_excel(path_or_buffer)

    # Normaliza nombres típicos (si vienen con unidades)
    rename_map = {
        "SiO2 (wt%)":"SiO2","TiO2 (wt%)":"TiO2","Al2O3 (wt%)":"Al2O3",
        "FeOt (wt%)":"FeOt",
        "MgO (wt%)":"MgO","CaO (wt%)":"CaO","Na2O (wt%)":"Na2O","K2O (wt%)":"K2O","P2O5 (wt%)":"P2O5",
        "Rb (ppm)":"Rb","Ba (ppm)":"Ba","Th (ppm)":"Th","Nb (ppm)":"Nb",
        "La (ppm)":"La","Zr (ppm)":"Zr","Y (ppm)":"Y","Yb (ppm)":"Yb",
    }
    df = df.rename(columns=rename_map)

    # fuerza numéricas en columnas conocidas
    for c in rename_map.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_ratios(df)
    return df


# ============================================================
# 2) MODELO: VARIABLES + PIPELINE
# ============================================================
def build_train_cols(df_train: pd.DataFrame) -> list[str]:
    core = [
        "SiO2","TiO2","Al2O3","FeOt","MgO","CaO",
        "Na2O","K2O","P2O5","Rb","Ba","Th","Nb",
        "La","Zr","Y","Yb"
    ]
    ratios = ["Nb_Y","Zr_Y","Th_Nb","La_Yb"]
    return [c for c in core + ratios if c in df_train.columns]


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
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


def add_prediction_columns(
    df_new: pd.DataFrame,
    model: Pipeline,
    train_cols: list[str],
    threshold_transicional: float = THRESHOLD_TRANSICIONAL,
    delta_th: float = DEFAULT_DELTA_TH,
    group_col_out: str = "Pred_Final",
) -> pd.DataFrame:
    """Añade predicción, probabilidades y ambigüedad a df_new."""
    df_new = df_new.copy()

    # asegurar columnas
    for c in train_cols:
        if c not in df_new.columns:
            df_new[c] = np.nan

    df_new = add_ratios(df_new)

    X_new = df_new[train_cols].copy()
    proba = model.predict_proba(X_new)
    pred = model.predict(X_new)
    classes = model.classes_

    df_new["Pred_Ambiente"] = pred
    df_new["Confianza"] = proba.max(axis=1)

    df_new[group_col_out] = np.where(
        df_new["Confianza"] >= float(threshold_transicional),
        df_new["Pred_Ambiente"],
        "Transicional"
    )

    # columnas proba_*
    for i, c in enumerate(classes):
        df_new[f"Proba_{c}"] = proba[:, i]

    # Top1/Top2 y Delta
    order = np.argsort(proba, axis=1)[:, ::-1]
    top1 = order[:, 0]
    top2 = order[:, 1]

    df_new["Top1_Clase"] = [classes[i] for i in top1]
    df_new["Top1_Prob"] = proba[np.arange(len(df_new)), top1]
    df_new["Top2_Clase"] = [classes[i] for i in top2]
    df_new["Top2_Prob"] = proba[np.arange(len(df_new)), top2]
    df_new["Delta_12"] = df_new["Top1_Prob"] - df_new["Top2_Prob"]
    df_new["Ambiguo"] = df_new["Delta_12"] < float(delta_th)

    return df_new


# ============================================================
# 3) GRÁFICAS
# ============================================================
def iter_groups(df: pd.DataFrame, group_col: Optional[str]) -> list[str]:
    if not group_col or group_col not in df.columns:
        return []
    groups = [str(g) for g in df[group_col].dropna().unique()]
    non_t = [g for g in groups if g.lower() != "transicional"]
    t = [g for g in groups if g.lower() == "transicional"]
    return sorted(non_t) + t


def pick_group_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["Pred_Final","Pred_Ambiente",LABEL_COL,"Ambiente","Grupo","Sample SubType","Sample Type"]:
        if c in df.columns:
            return c
    return None


def harker_suite_facets(df: pd.DataFrame, group_col: Optional[str], ncols: int = 3,
                       s: int = 12, alpha: float = 0.55) -> None:
    if "SiO2" not in df.columns:
        print("⚠️ Harker: falta SiO2.")
        return
    oxides = [c for c in ["TiO2","Al2O3","FeOt","MgO","CaO","Na2O","K2O","P2O5"] if c in df.columns]
    if not oxides:
        print("⚠️ Harker: faltan óxidos.")
        return

    n = len(oxides)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.2*nrows), squeeze=False)

    for i, ox in enumerate(oxides):
        ax = axes[i//ncols][i % ncols]
        dd = df.dropna(subset=["SiO2", ox]).copy()

        if group_col and group_col in dd.columns:
            for gname, g in dd.groupby(group_col):
                ax.scatter(g["SiO2"], g[ox], s=s, alpha=alpha, label=str(gname))
        else:
            ax.scatter(dd["SiO2"], dd[ox], s=s, alpha=alpha)

        ax.set_title(f"SiO2 vs {ox}", fontsize=10)
        ax.set_xlabel("SiO2")
        ax.set_ylabel(ox)

    for j in range(n, nrows*ncols):
        axes[j//ncols][j % ncols].axis("off")

    # leyenda afuera si hay grupos
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    plt.tight_layout()
    plt.show()


def tas_rock_type_simple(df: pd.DataFrame) -> Optional[str]:
    for c in ["TipoRoca","Tipo_Roca","RockType","Rock_Type","Roca","Tipo"]:
        if c in df.columns:
            return c
    return None


def plot_tas_fields_clean(ax) -> None:
    ax.set_xlim(35, 80)
    ax.set_ylim(0, 16)

    for xv in [41, 45, 52, 57, 63, 69]:
        ax.axvline(xv, linewidth=0.9, alpha=0.35)

    xs = np.array([41, 45, 52, 57, 63, 69, 77])
    ys = np.array([3.0, 5.0, 5.5, 6.0, 7.0, 8.0, 1.0])
    ax.plot(xs, ys, linewidth=1.2, alpha=0.85)

    ax.text(45.5, 1.8, "Basalto", fontsize=9)
    ax.text(55.0, 2.2, "Andesita\nbasáltica", fontsize=8, ha="center")
    ax.text(60.0, 2.6, "Andesita", fontsize=9, ha="center")
    ax.text(66.0, 4.3, "Dacita", fontsize=9, ha="center")
    ax.text(73.0, 6.2, "Riolita", fontsize=9, ha="center")

    ax.set_xlabel("SiO$_2$ (wt%)")
    ax.set_ylabel("Na$_2$O + K$_2$O (wt%)")


def tas_facets_by_rocktype(df: pd.DataFrame, group_col: Optional[str], ncols: int = 3,
                           s: int = 12, alpha: float = 0.55, max_points: int = 2500, seed: int = 42) -> None:
    need = {"SiO2","Na2O","K2O"}
    if not need.issubset(df.columns):
        print("⚠️ TAS: faltan SiO2, Na2O o K2O.")
        return

    dd = df.dropna(subset=["SiO2","Na2O","K2O"]).copy()
    dd["Na2OplusK2O"] = dd["Na2O"] + dd["K2O"]

    if not group_col or group_col not in dd.columns:
        # un solo panel si no hay grupos
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_tas_fields_clean(ax)
        ax.scatter(dd["SiO2"], dd["Na2OplusK2O"], s=s, alpha=alpha)
        plt.tight_layout()
        plt.show()
        return

    rock_col = tas_rock_type_simple(dd)
    groups = iter_groups(dd, group_col)
    if not groups:
        print("⚠️ TAS: no hay grupos.")
        return

    n = len(groups)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.4*nrows), squeeze=False)

    for i, gname in enumerate(groups):
        ax = axes[i//ncols][i % ncols]
        g = dd[dd[group_col].astype(str) == str(gname)].copy()
        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        plot_tas_fields_clean(ax)

        if rock_col is not None:
            for rname, gr in g.groupby(rock_col):
                ax.scatter(gr["SiO2"], gr["Na2OplusK2O"], s=s, alpha=alpha, label=str(rname), rasterized=True)
            ax.legend(fontsize=7, loc="best", frameon=True)
        else:
            ax.scatter(g["SiO2"], g["Na2OplusK2O"], s=s, alpha=alpha, rasterized=True)

        ax.set_title(str(gname), fontsize=9)

    for j in range(n, nrows*ncols):
        axes[j//ncols][j % ncols].axis("off")

    plt.tight_layout()
    plt.show()


def afm_facets(df: pd.DataFrame, group_col: Optional[str], ncols: int = 3,
               s: int = 12, alpha: float = 0.6, max_points: int = 2500, seed: int = 42) -> None:
    need = {"Na2O","K2O","FeOt","MgO"}
    if not need.issubset(df.columns):
        print("⚠️ AFM: faltan Na2O, K2O, FeOt o MgO.")
        return

    dd = df.copy()
    dd["A"] = dd["Na2O"] + dd["K2O"]
    dd["F"] = dd["FeOt"]
    dd["M"] = dd["MgO"]
    dd = dd.dropna(subset=["A","F","M"])
    dd = dd[(dd["A"]>0) & (dd["F"]>0) & (dd["M"]>0)].copy()
    if dd.empty:
        print("⚠️ AFM: no hay datos.")
        return

    ssum = dd[["A","F","M"]].sum(axis=1).replace(0, np.nan)
    dd["a"] = dd["A"]/ssum
    dd["f"] = dd["F"]/ssum
    dd["m"] = dd["M"]/ssum
    dd = dd.dropna(subset=["a","f","m"])

    SQ3_2 = np.sqrt(3)/2
    dd["x"] = dd["m"] + 0.5*dd["f"]
    dd["y"] = dd["f"] * SQ3_2

    groups = iter_groups(dd, group_col)
    if not groups:
        groups = ["(sin grupos)"]

    n = len(groups)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows), squeeze=False)

    for i, gname in enumerate(groups):
        ax = axes[i//ncols][i % ncols]
        if gname == "(sin grupos)" or (not group_col) or group_col not in dd.columns:
            g = dd.copy()
        else:
            g = dd[dd[group_col].astype(str) == str(gname)].copy()

        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        ax.plot([0,1,0.5,0],[0,0,SQ3_2,0], linewidth=1)
        ax.scatter(g["x"], g["y"], s=s, alpha=alpha)
        ax.set_title(str(gname), fontsize=9)
        ax.axis("off")

    for j in range(n, nrows*ncols):
        axes[j//ncols][j % ncols].axis("off")

    plt.tight_layout()
    plt.show()


def spider_prepare(df: pd.DataFrame, group_col: Optional[str]):
    elems = [e for e in ["Rb","Ba","Th","Nb","La","Ce","Nd","Sm","Zr","Hf","Y","Yb","Lu","Sr"] if e in df.columns]
    if len(elems) < 6:
        return None, None, None
    cols = elems + ([group_col] if group_col and group_col in df.columns else [])
    dd = df[cols].copy()
    dd[elems] = dd[elems].apply(pd.to_numeric, errors="coerce")
    dd[elems] = dd[elems].mask(dd[elems] <= 0, np.nan)
    dd = dd.dropna(how="all", subset=elems)
    return dd, elems, np.arange(len(elems))


def spider_all_classes(df: pd.DataFrame, group_col: Optional[str]) -> None:
    dd, elems, x = spider_prepare(df, group_col)
    if dd is None:
        print("⚠️ Spider: no hay suficientes elementos (mín. 6).")
        return

    plt.figure(figsize=(8,4))
    if group_col and group_col in dd.columns:
        for k, g in dd.groupby(group_col):
            med = g[elems].median(skipna=True)
            plt.plot(x, med.values, marker="o", linewidth=1, label=str(k))
        plt.legend(fontsize=8, loc="best")
    else:
        med = dd[elems].median(skipna=True)
        plt.plot(x, med.values, marker="o", linewidth=1)

    plt.yscale("log")
    plt.xticks(x, elems, rotation=45, ha="right")
    plt.title("Spider (medianas) — escala log (sin normalización)")
    plt.ylabel("Concentración")
    plt.tight_layout()
    plt.show()


def spider_facets(df: pd.DataFrame, group_col: Optional[str], ncols: int = 3) -> None:
    dd, elems, x = spider_prepare(df, group_col)
    if dd is None:
        print("⚠️ Spider: no hay suficientes elementos (mín. 6).")
        return

    groups = iter_groups(dd, group_col)
    if not groups:
        groups = ["(sin grupos)"]

    n = len(groups)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4*ncols, 3.6*nrows), squeeze=False)

    for i, gname in enumerate(groups):
        ax = axes[i//ncols][i % ncols]
        if gname == "(sin grupos)" or (not group_col) or group_col not in dd.columns:
            g = dd.copy()
        else:
            g = dd[dd[group_col].astype(str) == str(gname)]

        med = g[elems].median(skipna=True)
        ax.plot(x, med.values, marker="o", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(elems, rotation=45, ha="right", fontsize=7)
        ax.set_title(str(gname), fontsize=9)

    for j in range(n, nrows*ncols):
        axes[j//ncols][j % ncols].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Coordenadas: detección robusta de lat/lon
# ============================================================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def detect_lat_lon_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat_keys = ["lat", "latitude", "latitud", "y", "north", "northing"]
    lon_keys = ["lon", "lng", "long", "longitude", "longitud", "x", "east", "easting"]

    lat = next((c for c in df.columns if any(k in _norm(c) for k in lat_keys)), None)
    lon = next((c for c in df.columns if any(k in _norm(c) for k in lon_keys)), None)
    return lat, lon
