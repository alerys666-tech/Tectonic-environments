
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

def _safe_numeric(series: pd.Series) -> pd.Series:
    # Convierte texto con comas/espacios a numérico
    s = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _scatter_simple(ax, x, y, s=12, alpha=0.55):
    ax.scatter(x, y, s=s, alpha=alpha, rasterized=True)
    
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


def plot_tas_fields_reference(ax):
    ax.set_xlim(37, 77)
    ax.set_ylim(0, 16)

   
    for xv in [41, 45, 49, 52, 57, 63, 69, 73]:
        ax.axvline(xv, linewidth=0.8, alpha=0.25)

   
    def draw_poly(xy, lw=1.2):
        xs, ys = zip(*xy)
        ax.plot(xs, ys, linewidth=lw, alpha=0.95)

    # -------------------------
    # Campo: Subalcalino (principal)
    # -------------------------
    # Basalt / Andesites / Dacite / Rhyolite (simplificado pero “como figura”)
    # Basalt: ~45-52, bajo
    draw_poly([(45, 0.0), (52, 0.0), (52, 5.0), (45, 5.0), (45, 0.0)])
    ax.text(48.5, 2.2, "basalt", fontsize=9, ha="center")

    # Basaltic andesite: 52-57
    draw_poly([(52, 0.0), (57, 0.0), (57, 5.8), (52, 5.8), (52, 0.0)])
    ax.text(54.5, 2.4, "basaltic\nandesite", fontsize=8, ha="center")

    # Andesite: 57-63
    draw_poly([(57, 0.0), (63, 0.0), (63, 6.5), (57, 6.5), (57, 0.0)])
    ax.text(60.0, 3.0, "andesite", fontsize=9, ha="center")

    # Dacite: 63-69
    draw_poly([(63, 0.0), (69, 0.0), (69, 7.5), (63, 7.5), (63, 0.0)])
    ax.text(66.0, 4.0, "dacite", fontsize=9, ha="center")

    # Rhyolite: >69 (cuña)
    draw_poly([(69, 0.0), (77, 0.0), (77, 16), (73, 8.0), (69, 7.5), (69, 0.0)])
    ax.text(74.5, 8.7, "rhyolite", fontsize=9, ha="center")

    # Picrobasalt: ~41-45, muy bajo
    draw_poly([(41, 0.0), (45, 0.0), (45, 3.0), (41, 3.0), (41, 0.0)])
    ax.text(43.0, 1.3, "picro-\nbasalt", fontsize=8, ha="center")

    # -------------------------
    # Campo: Alcalino (izquierda-media)
    # basanite / tephrite, trachybasalt, basaltic trachyandesite
    # -------------------------
    # Basanite/Tephrite (ol<10) ~41-45, 3-7
    draw_poly([(41, 3.0), (45, 3.0), (45, 7.0), (41, 7.0), (41, 3.0)])
    ax.text(43.0, 5.0, "basanite\n(ol<10%)\ntephrite\n(ol<10%)", fontsize=7, ha="center")

    # Trachybasalt ~45-52, 5-9
    draw_poly([(45, 5.0), (52, 5.0), (52, 9.0), (45, 9.0), (45, 5.0)])
    ax.text(48.5, 7.0, "trachy-\nbasalt", fontsize=8, ha="center")

    # Basaltic trachyandesite ~52-57, 5.8-10
    draw_poly([(52, 5.8), (57, 5.8), (57, 10.0), (52, 10.0), (52, 5.8)])
    ax.text(54.5, 8.0, "basaltic\ntrachy-\nandesite", fontsize=7, ha="center")

    # Trachyandesite ~57-63, 6.5-11
    draw_poly([(57, 6.5), (63, 6.5), (63, 11.0), (57, 11.0), (57, 6.5)])
    ax.text(60.0, 8.7, "trachy-\nandesite", fontsize=8, ha="center")

    # Trachydacite ~63-69, 7.5-12
    draw_poly([(63, 7.5), (69, 7.5), (69, 12.0), (63, 12.0), (63, 7.5)])
    ax.text(66.0, 9.8, "trachy-\ndacite", fontsize=8, ha="center")

    # Trachyte (q<20%) ~69-73, 8-13
    draw_poly([(69, 8.0), (73, 8.0), (73, 13.0), (69, 13.0), (69, 8.0)])
    ax.text(71.0, 10.7, "trachyte\n(q<20%)", fontsize=8, ha="center")

    # -------------------------
    # Campos: Fonolíticos (parte alta central)
    # -------------------------
    # Phonolite (tope) ~52-63, 12-16
    draw_poly([(52, 12.0), (63, 12.0), (63, 16.0), (52, 16.0), (52, 12.0)])
    ax.text(57.5, 14.4, "phonolite", fontsize=9, ha="center")

    # Tephriphonolite ~49-57, 10-12
    draw_poly([(49, 10.0), (57, 10.0), (57, 12.0), (49, 12.0), (49, 10.0)])
    ax.text(53.0, 11.0, "tephri-\nphonolite", fontsize=8, ha="center")

    # Phonotephrite ~45-52, 9-11
    draw_poly([(45, 9.0), (52, 9.0), (52, 11.0), (45, 11.0), (45, 9.0)])
    ax.text(48.5, 10.0, "phono-\ntephrite", fontsize=8, ha="center")

    # Foidite (gran campo alto izquierdo) ~37-45, 11-16
    draw_poly([(37, 11.0), (45, 11.0), (45, 16.0), (37, 16.0), (37, 11.0)])
    ax.text(41.0, 13.5, "foidite", fontsize=9, ha="center")

    ax.set_xlabel("Wt. % SiO$_2$")
    ax.set_ylabel("Wt. % Na$_2$O = K$_2$O")
    ax.set_title("TAS (campos clásicos, estilo referencia)")
    ax.grid(True, alpha=0.12)


def tas_rock_type_simple(df: pd.DataFrame) -> Optional[str]:
 
    for c in ["TipoRoca", "Tipo_Roca", "RockType", "Rock_Type", "Roca", "Tipo", "Lithology", "Litologia"]:
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


def tas_facets_by_rocktype(df: pd.DataFrame, group_col: Optional[str], ncols: int = 3,
                           s: int = 12, alpha: float = 0.55, max_points: int = 2500, seed: int = 42) -> None:

    need = {"SiO2","Na2O","K2O"}
    if not need.issubset(df.columns):
        print("⚠️ TAS: faltan SiO2, Na2O o K2O.")
        return

    dd = df.dropna(subset=["SiO2","Na2O","K2O"]).copy()
    dd["Na2OplusK2O"] = dd["Na2O"] + dd["K2O"]

    if not group_col or group_col not in dd.columns:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_tas_fields_reference(ax)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4*ncols, 3.6*nrows), squeeze=False)

    for i, gname in enumerate(groups):
        ax = axes[i//ncols][i % ncols]
        g = dd[dd[group_col].astype(str) == str(gname)].copy()
        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        plot_tas_fields_reference(ax)
        
        ax.scatter(
            g["SiO2"],
            g["Na2OplusK2O"],
            s=s,
            alpha=alpha,
            color="black",
            rasterized=True
        )

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
def plot_k2o_sio2_series_fields(ax):
    """
    Campos aproximados clásicos para K2O vs SiO2:
    - Toleítica
    - Calco-alcalina
    - Calco-alcalina alto K
    - Shoshonítica
    (Líneas aproximadas ampliamente usadas en petrología)
    """
    ax.set_xlim(45, 80)
    ax.set_ylim(0, 7)

    # Líneas aproximadas separadoras (se ven como en tu figura)
    # Toleítica / Calco-alcalina
    x = np.array([45, 50, 55, 60, 65, 70, 75, 80])
    y1 = np.array([0.2, 0.35, 0.55, 0.8, 1.05, 1.25, 1.45, 1.6])  # toleítica / calcoalcalina

    # Calco-alcalina / Alto-K calcoalcalina
    y2 = np.array([0.6, 0.9, 1.25, 1.65, 2.05, 2.45, 2.85, 3.2])

    # Alto-K / Shoshonítica
    y3 = np.array([1.6, 2.2, 3.2, 3.6, 4.1, 4.6, 5.1, 5.6])

    ax.plot(x, y1, linewidth=1.2)
    ax.plot(x, y2, linewidth=1.2)
    ax.plot(x, y3, linewidth=1.2)

    # Etiquetas
    ax.text(73, 0.9, "serie Toleítica", fontsize=9, ha="right")
    ax.text(73, 2.0, "serie Calco-alcalina", fontsize=9, ha="right")
    ax.text(76, 3.7, "serie Calco-alcalina\ncon alto K", fontsize=9, ha="right")
    ax.text(55, 4.6, "serie Shoshonítica", fontsize=9, ha="center")

    ax.set_xlabel("SiO$_2$ (wt%)")
    ax.set_ylabel("K$_2$O (wt%)")
    ax.set_title("K$_2$O vs SiO$_2$ (series magmáticas)")
    ax.grid(True, alpha=0.15)

def k2o_sio2_series_facets(
    df: pd.DataFrame,
    group_col: Optional[str],
    ncols: int = 3,
    s: int = 12,
    alpha: float = 0.55,
    max_points: int = 2500,
    seed: int = 42
) -> None:
    need = {"SiO2","K2O"}
    if not need.issubset(df.columns):
        print("⚠️ K2O-SiO2: faltan SiO2 o K2O.")
        return

    dd = df.copy()
    dd["SiO2"] = _safe_numeric(dd["SiO2"])
    dd["K2O"]  = _safe_numeric(dd["K2O"])
    dd = dd.dropna(subset=["SiO2","K2O"])
    if dd.empty:
        print("⚠️ K2O-SiO2: sin datos válidos.")
        return

    if (not group_col) or (group_col not in dd.columns):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        plot_k2o_sio2_series_fields(ax)
        _scatter_simple(ax, dd["SiO2"], dd["K2O"], s=s, alpha=alpha)
        plt.tight_layout(); plt.show()
        return

    groups = iter_groups(dd, group_col)
    n = len(groups)
    if n == 0:
        print("⚠️ K2O-SiO2: no hay grupos.")
        return

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4*ncols, 3.6*nrows), squeeze=False)

    for i, gname in enumerate(groups):
        ax = axes[i//ncols][i % ncols]
        g = dd[dd[group_col].astype(str) == str(gname)].copy()
        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        plot_k2o_sio2_series_fields(ax)
        _scatter_simple(ax, g["SiO2"], g["K2O"], s=s, alpha=alpha)
        ax.set_title(str(gname), fontsize=9)

    for j in range(n, nrows*ncols):
        axes[j//ncols][j % ncols].axis("off")

    plt.tight_layout(); plt.show()


# ============================================================
# 2) Shand diagram: A/NK vs A/CNK (peralcalina, metaluminosa, peraluminosa)
# ============================================================
def plot_shand_fields(ax):

    ax.set_xlim(0.5, 2.0)
    ax.set_ylim(0.5, 7.0)

    # Línea vertical A/CNK = 1
    ax.axvline(1.0, linewidth=1.2, alpha=0.8)
    # Línea horizontal A/NK = 1
    ax.axhline(1.0, linewidth=1.2, alpha=0.8)

    # Línea guía típica (inclinada) a veces se dibuja (opcional)
    x = np.array([0.5, 2.0])
    y = 0.6 + 0.7*(x - 0.5)  # línea suave estilo tu figura
    ax.plot(x, y, linestyle="--", linewidth=1.0, alpha=0.7)

    ax.text(0.75, 0.75, "Peralcalina", fontsize=9, ha="center", alpha=0.9)
    ax.text(0.75, 6.2, "Metaluminosa", fontsize=9, ha="center", alpha=0.9)
    ax.text(1.45, 6.2, "Peraluminosa", fontsize=9, ha="center", alpha=0.9)

    ax.set_xlabel("A/CNK = Al$_2$O$_3$ / (CaO + Na$_2$O + K$_2$O)")
    ax.set_ylabel("A/NK = Al$_2$O$_3$ / (Na$_2$O + K$_2$O)")
    ax.set_title("Shand: A/NK vs A/CNK")
    ax.grid(True, alpha=0.15)

def shand_facets(
    df: pd.DataFrame,
    group_col: Optional[str],
    ncols: int = 3,
    s: int = 12,
    alpha: float = 0.55,
    max_points: int = 2500,
    seed: int = 42
) -> None:
    need = {"Al2O3","CaO","Na2O","K2O"}
    if not need.issubset(df.columns):
        print("⚠️ Shand: faltan Al2O3, CaO, Na2O o K2O.")
        return

    dd = df.copy()
    for c in ["Al2O3","CaO","Na2O","K2O"]:
        dd[c] = _safe_numeric(dd[c])

    dd = dd.dropna(subset=["Al2O3","CaO","Na2O","K2O"]).copy()
    dd = dd[(dd["Al2O3"] > 0) & ((dd["Na2O"] + dd["K2O"]) > 0) & ((dd["CaO"] + dd["Na2O"] + dd["K2O"]) > 0)]
    if dd.empty:
        print("⚠️ Shand: sin datos válidos.")
        return

    dd["A_NK"]  = dd["Al2O3"] / (dd["Na2O"] + dd["K2O"])
    dd["A_CNK"] = dd["Al2O3"] / (dd["CaO"] + dd["Na2O"] + dd["K2O"])

    # Limitar rangos para plot limpio
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna(subset=["A_NK","A_CNK"])
    dd = dd[(dd["A_CNK"].between(0.5, 2.0)) & (dd["A_NK"].between(0.5, 7.0))]
    if dd.empty:
        print("⚠️ Shand: todo quedó fuera de rango (revisar óxidos / unidades).")
        return

    if (not group_col) or (group_col not in dd.columns):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        plot_shand_fields(ax)
        _scatter_simple(ax, dd["A_CNK"], dd["A_NK"], s=s, alpha=alpha)
        plt.tight_layout(); plt.show()
        return

    groups = iter_groups(dd, group_col)
    n = len(groups)
    if n == 0:
        print("⚠️ Shand: no hay grupos.")
        return

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.7*nrows), squeeze=False)

    for i, gname in enumerate(groups):
        ax = axes[i//ncols][i % ncols]
        g = dd[dd[group_col].astype(str) == str(gname)].copy()
        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)

        plot_shand_fields(ax)
        _scatter_simple(ax, g["A_CNK"], g["A_NK"], s=s, alpha=alpha)
        ax.set_title(str(gname), fontsize=9)

    for j in range(n, nrows*ncols):
        axes[j//ncols][j % ncols].axis("off")

    plt.tight_layout(); plt.show()

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
