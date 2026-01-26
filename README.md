# Tectonic Environment Classifier (Geoquímica + ML)

Aplicación sencilla para **predecir ambientes tectónicos** a partir de datos geoquímicos (Excel/CSV) usando un modelo entrenado (ExtraTrees + imputer) guardado con `joblib`.

## Archivos principales
- `app.py` : App web (Streamlit) para subir un archivo y obtener predicciones + gráficos.
- `geo_model.py` : Funciones de carga, preparación, predicción y gráficos.
- `requirements.txt` : Dependencias.

## Estructura recomendada del repo
```
.
├─ app.py
├─ geo_model.py
├─ requirements.txt
├─ README.md
└─ models/
   ├─ modelo_et_tectonica.pkl
   ├─ imputer_median.pkl
   └─ feature_cols.pkl
```

> Coloca tus artefactos entrenados dentro de `models/` con esos nombres (o cambia las rutas en `geo_model.py`).

## Cómo ejecutar (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Formato de entrada
- Archivo `.xlsx`, `.xls` o `.csv`
- Debe contener (al menos parcialmente) las columnas que el modelo espera (`feature_cols.pkl`).
- Si existen `Nb`, `Yb`, `Th`, `Zr`, `La`, se calculan ratios (`Nb_Yb`, `Th_Yb`, `Zr_Nb`, `La_Yb`).

## Salidas
- Tabla con `Pred`, `Confianza`, `Delta_12`, `Pred_Final`
- Descarga de Excel con predicciones
- Gráficos: histograma de clases, AFM ternario (si hay Na2O/K2O/FeOt/MgO), Harker (SiO2 vs óxidos)

## Notas
- `Pred_Final` marca **Indeterminado** si:
  - `Confianza < 0.45` o
  - `Delta_12 < 0.07`
  Puedes ajustar esos umbrales en `app.py`.
