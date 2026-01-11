import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import geo_model
except Exception as e:
    st.error("❌ Error cargando geo_model.py")
    st.exception(e)
    st.stop()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

st.set_page_config(
    page_title="IA Geoquímica – Ambientes Tectónicos",
    page_icon="🌋💻",
    layout="wide"
)

st.title("🌋💻 Predicción de Ambientes Tectónicos con IA")
st.caption("Modelo de Machine Learning aplicado a datos geoquímicos volcánicos")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Parámetros")
    delta_th = st.slider("Umbral ambigüedad (Delta Top1–Top2)", 0.01, 0.5, 0.15, 0.01)
    thr_trans = st.slider("Umbral Transicional (Confianza)", 0.1, 0.95, geo_model.THRESHOLD_TRANSICIONAL, 0.01)

# =========================
# UPLOAD
# =========================
col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader("📄 Subir archivo de entrenamiento (con columna Ambiente_Tectonico)", type=["xlsx","csv"])
with col2:
    new_file = st.file_uploader("📄 Subir archivo nuevo para predecir", type=["xlsx","csv"])

if not train_file or not new_file:
    st.info("⬆️ Sube ambos archivos para ejecutar el modelo.")
    st.stop()

# =========================
# ENTRENAMIENTO
# =========================
df_train = geo_model.read_any(train_file)

if geo_model.LABEL_COL not in df_train.columns:
    st.error("❌ El archivo de entrenamiento no tiene la columna Ambiente_Tectonico")
    st.stop()

train_cols = geo_model.build_train_cols(df_train)

df_train = df_train.dropna(subset=[geo_model.LABEL_COL])
X = df_train[train_cols]
y = df_train[geo_model.LABEL_COL].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = geo_model.build_model()
model.fit(X_tr, y_tr)

# =========================
# VALIDACIÓN
# =========================
y_pred_te = model.predict(X_te)
acc_tr = accuracy_score(y_tr, model.predict(X_tr))
acc_te = accuracy_score(y_te, y_pred_te)

tab_res, tab_val, tab_pred, tab_classic, tab_map = st.tabs(
    ["📌 Resumen", "✅ Validación", "🔮 Predicción", "📈 Diagramas clásicos", "🗺️ Mapa"]
)

with tab_res:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy Train", f"{acc_tr:.3f}")
    c2.metric("Accuracy Test", f"{acc_te:.3f}")
    c3.metric("Variables", len(train_cols))

    st.write("**Variables usadas:**")
    st.code(", ".join(train_cols))

with tab_val:
    st.text(classification_report(y_te, y_pred_te))
    cm = confusion_matrix(y_te, y_pred_te, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(ax=ax, cmap="Blues")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close()

# =========================
# PREDICCIÓN
# =========================
df_new = geo_model.read_any(new_file)

for c in train_cols:
    if c not in df_new.columns:
        df_new[c] = np.nan

df_new = geo_model.add_ratios(df_new)

X_new = df_new[train_cols]
proba = model.predict_proba(X_new)
pred = model.predict(X_new)
classes = model.classes_

df_new["Pred_Ambiente"] = pred
df_new["Confianza"] = proba.max(axis=1)

df_new["Pred_Final"] = np.where(
    df_new["Confianza"] >= thr_trans,
    df_new["Pred_Ambiente"],
    "Transicional"
)

# Probabilidades
for i, c in enumerate(classes):
    df_new[f"Proba_{c}"] = proba[:, i]

# Ambigüedad
order = np.argsort(proba, axis=1)[:, ::-1]
df_new["Top1_Clase"] = classes[order[:,0]]
df_new["Top2_Clase"] = classes[order[:,1]]
df_new["Delta_12"] = proba[np.arange(len(df_new)), order[:,0]] - proba[np.arange(len(df_new)), order[:,1]]
df_new["Ambiguo"] = df_new["Delta_12"] < delta_th

with tab_pred:
    st.metric("Ambiguos", int(df_new["Ambiguo"].sum()))
    st.dataframe(df_new.head(40), use_container_width=True)

# =========================
# DIAGRAMAS CLÁSICOS
# =========================
with tab_classic:
    group_col = geo_model.pick_group_col(df_new)

    if st.checkbox("TAS (Solo rocas volcánicas"):
        geo_model.tas_facets_by_rocktype(df_new, group_col)
        st.pyplot(plt.gcf()); plt.close()

    if st.checkbox("AFM"):
        geo_model.afm_facets(df_new, group_col)
        st.pyplot(plt.gcf()); plt.close()

    if st.checkbox("Spider"):
        geo_model.spider_all_classes(df_new, group_col)
        st.pyplot(plt.gcf()); plt.close()
    if st.checkbox("Harker"):
        geo_model.harker_suite_facets(df_new, group_col)
        st.pyplot(plt.gcf()); plt.close()
# =========================
# MAPA
# =========================
# --- MAPA (robusto) ---
lat, lon = geo_model.detect_lat_lon_cols(df_new)

if lat and lon:
    ddm = df_new.copy()

    # convertir a numérico (arregla comas, grados, texto raro)
    ddm[lat] = (
        ddm[lat].astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("°", "", regex=False)
        .str.strip()
    )
    ddm[lon] = (
        ddm[lon].astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("°", "", regex=False)
        .str.strip()
    )

    ddm[lat] = pd.to_numeric(ddm[lat], errors="coerce")
    ddm[lon] = pd.to_numeric(ddm[lon], errors="coerce")

    # filtrar NaN y rangos válidos lat/lon
    ddm = ddm.dropna(subset=[lat, lon]).copy()
    ddm = ddm[(ddm[lat].between(-90, 90)) & (ddm[lon].between(-180, 180))]

    if ddm.empty:
        st.warning("Detecté columnas de coordenadas, pero no hay lat/lon válidos (pueden ser UTM o texto).")
    else:
        st.map(ddm.rename(columns={lat: "lat", lon: "lon"})[["lat", "lon"]])
else:
    st.info("No se detectaron columnas de latitud/longitud.")
