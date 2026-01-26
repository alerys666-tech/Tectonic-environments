import io
import pandas as pd
import streamlit as st

from geo_model import (
    load_artifacts,
    read_any,
    predict_dataframe,
    fig_histogram,
    fig_afm_ternary,
    fig_harker,
)

st.set_page_config(page_title="Ambientes tectónicos (Geoquímica + ML)", layout="wide")

st.title("Predicción de ambientes tectónicos (Geoquímica + ML)")
st.write("Sube un archivo **Excel/CSV** y obtén predicciones + gráficos.")

with st.sidebar:
    st.header("Modelo")
    models_dir = st.text_input("Carpeta de modelos", value="models")
    conf_min = st.slider("Umbral mínimo de confianza", 0.0, 1.0, 0.45, 0.01)
    delta_min = st.slider("Umbral mínimo Delta_12", 0.0, 1.0, 0.07, 0.01)
    st.caption("Si Confianza < umbral o Delta_12 < umbral ⇒ Indeterminado.")

uploaded = st.file_uploader("Archivo a predecir (.xlsx, .xls, .csv)", type=["xlsx","xls","csv"])

@st.cache_resource
def _load(models_dir: str):
    return load_artifacts(models_dir)

if uploaded is None:
    st.info("Sube un archivo para empezar.")
    st.stop()

# Cargar modelo
try:
    model, imputer, feature_cols = _load(models_dir)
    st.success(f"Modelo cargado. Features: {len(feature_cols)} | Clases: {list(model.classes_)}")
except Exception as e:
    st.error(f"No se pudo cargar el modelo desde '{models_dir}'.\n\n{e}")
    st.stop()

# Leer datos
try:
    df_in = read_any(uploaded)
except Exception as e:
    st.error(f"No se pudo leer el archivo.\n\n{e}")
    st.stop()

st.subheader("Vista previa del archivo")
st.dataframe(df_in.head(20))

# Predecir
df_pred = predict_dataframe(df_in, model, imputer, feature_cols, conf_min=conf_min, delta_min=delta_min)

st.subheader("Resultados")
st.write("Conteo por clase (Pred_Final):")
st.dataframe(df_pred["Pred_Final"].value_counts(dropna=False).rename("conteo").to_frame())

st.dataframe(df_pred.head(50))

# Descarga Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_pred.to_excel(writer, index=False, sheet_name="Predicciones")
output.seek(0)

st.download_button(
    "Descargar Excel con predicciones",
    data=output,
    file_name="predicciones_archivo_nuevo.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Gráficos
col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma")
    fig = fig_histogram(df_pred)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("AFM ternario (si aplica)")
    fig_afm = fig_afm_ternary(df_pred)
    if fig_afm is None:
        st.info("No se generó AFM (faltan columnas Na2O/K2O/FeOt/MgO o hay pocos datos válidos).")
    else:
        st.pyplot(fig_afm, clear_figure=True)

st.subheader("Harker (SiO2 vs óxidos)")
figs = fig_harker(df_pred)
if len(figs) == 0:
    st.info("No hay columnas suficientes para Harker (requiere SiO2 y al menos un óxido).")
else:
    for f in figs:
        st.pyplot(f, clear_figure=True)
