import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# IO + normalización
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Limpieza básica: espacios y subíndices químicos
    df = df.copy()
    df.columns = [
        str(c).strip()
              .replace("₂","2")
              .replace("₃","3")
              .replace("₅","5")
              .replace("₄","4")
        for c in df.columns
    ]
    return df

def read_any(path_or_buf) -> pd.DataFrame:
    """Lee CSV/Excel desde ruta (str/Path) o buffer (BytesIO)."""
    if hasattr(path_or_buf, "read"):
        # buffer
        name = getattr(path_or_buf, "name", "")
        ext = name.split(".")[-1].lower() if "." in name else ""
        if ext in ["xlsx","xls"]:
            df = pd.read_excel(path_or_buf)
        else:
            df = pd.read_csv(path_or_buf)
    else:
        path = str(path_or_buf)
        ext = path.split(".")[-1].lower()
        if ext in ["xlsx","xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    return normalize_columns(df)

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ("Nb" in df.columns) and ("Yb" in df.columns):
        df["Nb_Yb"] = safe_div(df["Nb"], df["Yb"])
    if ("Th" in df.columns) and ("Yb" in df.columns):
        df["Th_Yb"] = safe_div(df["Th"], df["Yb"])
    if ("Zr" in df.columns) and ("Nb" in df.columns):
        df["Zr_Nb"] = safe_div(df["Zr"], df["Nb"])
    if ("La" in df.columns) and ("Yb" in df.columns):
        df["La_Yb"] = safe_div(df["La"], df["Yb"])
    return df

# -----------------------------
# Modelo
# -----------------------------
def load_artifacts(models_dir: str | Path = "models"):
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "modelo_et_tectonica.pkl")
    imputer = joblib.load(models_dir / "imputer_median.pkl")
    feature_cols = joblib.load(models_dir / "feature_cols.pkl")
    return model, imputer, feature_cols

def predict_dataframe(
    df: pd.DataFrame,
    model,
    imputer,
    feature_cols,
    conf_min: float = 0.45,
    delta_min: float = 0.07,
):
    df0 = add_ratios(normalize_columns(df))
    # asegurar columnas
    for c in feature_cols:
        if c not in df0.columns:
            df0[c] = np.nan

    X = df0[feature_cols].apply(pd.to_numeric, errors="coerce")
    ok_pred = X.notna().any(axis=1)

    out = df0.copy()
    out["Pred"] = np.nan
    out["Confianza"] = np.nan
    out["Delta_12"] = np.nan

    if ok_pred.sum() > 0:
        X_sub = X.loc[ok_pred]
        X_imp = imputer.transform(X_sub)

        proba = model.predict_proba(X_imp)
        pred = model.predict(X_imp)

        conf = proba.max(axis=1)
        top2 = np.sort(proba, axis=1)[:, -2]
        delta = conf - top2

        out.loc[ok_pred, "Pred"] = pred
        out.loc[ok_pred, "Confianza"] = conf
        out.loc[ok_pred, "Delta_12"] = delta

    # Pred_Final solo donde hay predicción
    out["Pred_Final"] = out["Pred"]
    mask = out["Confianza"].notna()
    out.loc[mask, "Pred_Final"] = np.where(
        (out.loc[mask, "Confianza"] < conf_min) | (out.loc[mask, "Delta_12"] < delta_min),
        "Indeterminado",
        out.loc[mask, "Pred"],
    )
    return out

# -----------------------------
# Gráficos (matplotlib figs)
# -----------------------------
def fig_histogram(df_pred: pd.DataFrame, col: str = "Pred_Final"):
    fig, ax = plt.subplots(figsize=(7, 4))
    df_pred[col].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Histograma de ambientes predichos")
    ax.set_xlabel("Ambiente")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig

def fig_afm_ternary(df_pred: pd.DataFrame, class_col: str = "Pred_Final"):
    need = all(c in df_pred.columns for c in ["Na2O","K2O","FeOt","MgO"])
    if not need:
        return None

    Na2O = pd.to_numeric(df_pred["Na2O"], errors="coerce")
    K2O  = pd.to_numeric(df_pred["K2O"],  errors="coerce")
    FeOt = pd.to_numeric(df_pred["FeOt"], errors="coerce")
    MgO  = pd.to_numeric(df_pred["MgO"],  errors="coerce")

    A = Na2O + K2O
    F = FeOt
    M = MgO
    S = A + F + M
    ok = A.notna() & F.notna() & M.notna() & (S > 0)

    if ok.sum() < 5:
        return None

    A = (A[ok] / S[ok])
    F = (F[ok] / S[ok])
    M = (M[ok] / S[ok])
    labs = df_pred.loc[ok, class_col].astype(str)

    h = np.sqrt(3) / 2
    x = M + 0.5 * F
    y = h * F

    fig, ax = plt.subplots(figsize=(6, 5))
    tri = np.array([[0,0],[1,0],[0.5,h],[0,0]])
    ax.plot(tri[:,0], tri[:,1], color="black", linewidth=3)

    # curva roja (plantilla visual)
    t = np.linspace(0, 1, 200)
    x_curve = 0.18 + 0.64 * t
    y_curve = 0.28 + 0.08 * np.sin(np.pi * t)
    y_max = -np.sqrt(3) * np.abs(x_curve - 0.5) + h
    y_curve = np.minimum(y_curve, y_max - 0.01)
    ax.plot(x_curve, y_curve, color="#7A0C0C", linewidth=3)

    ax.text(0.5, h + 0.03, "F", ha="center", va="bottom", fontsize=12)
    ax.text(-0.02, -0.02, "A", ha="right", va="top", fontsize=12)
    ax.text(1.02, -0.02, "M", ha="left", va="top", fontsize=12)
    ax.text(0.50, 0.55, "Series toleíticas", ha="center", va="center", fontsize=11)
    ax.text(0.50, 0.10, "Series calcoalcalinas", ha="center", va="center", fontsize=11)

    for lab in labs.unique():
        idx = labs.eq(lab).values
        ax.scatter(x.values[idx], y.values[idx], s=18, alpha=0.65, label=lab)

    ax.set_title("AFM ternario (plantilla)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, h + 0.08)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    return fig

def fig_harker(df_pred: pd.DataFrame, class_col: str = "Pred_Final", xcol: str = "SiO2"):
    ycols = [c for c in ["TiO2","Al2O3","FeOt","MgO","CaO","Na2O","K2O","P2O5"] if c in df_pred.columns]
    if xcol not in df_pred.columns or len(ycols) == 0:
        return []

    figs = []
    for ycol in ycols:
        fig, ax = plt.subplots(figsize=(7, 5))
        for lab, g in df_pred.groupby(class_col):
            if str(lab) == "Indeterminado":
                continue
            x = pd.to_numeric(g[xcol], errors="coerce")
            y = pd.to_numeric(g[ycol], errors="coerce")
            ax.scatter(x, y, s=14, alpha=0.35, label=str(lab))
        ax.set_title(f"Harker: {xcol} vs {ycol}")
        ax.set_xlabel("SiO2 (wt%)")
        ax.set_ylabel(f"{ycol} (wt%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        figs.append(fig)
    return figs
