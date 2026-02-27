import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. Carga de Artefactos del Pipeline y Modelos ---

@st.cache_resource
def load_pipeline_artefacts():
    pipeline_artefacts_path = 'pipeline_artefacts.joblib'
    pipeline_artefacts = joblib.load(pipeline_artefacts_path)

    scaler = pipeline_artefacts['scaler']
    encoders = pipeline_artefacts['encoders']
    independent_vars = pipeline_artefacts['independent_vars']
    target_score_cols = pipeline_artefacts['score_cols']
    asset_cols = pipeline_artefacts['asset_cols'] # For INDICE_BIENES

    loaded_models = {}
    for target in target_score_cols:
        model_filename = f"icfes_mejor_{target.lower()}.joblib"
        if os.path.exists(model_filename):
            model_data = joblib.load(model_filename)
            loaded_models[target] = model_data['modelo_fit']
        else:
            st.error(f"Error: Modelo para {target} no encontrado: {model_filename}")
            st.stop()

    return scaler, encoders, independent_vars, target_score_cols, loaded_models, asset_cols

scaler, encoders, independent_vars, target_score_cols, loaded_models, asset_cols = load_pipeline_artefacts()

# --- 2. Funciones de Preprocesamiento para la Entrada del Usuario ---
def preprocess_input(user_input_data: dict) -> pd.DataFrame:
    # Create a DataFrame from the user input
    df_single_row = pd.DataFrame([user_input_data])

    # Reconstruct INDICE_BIENES
    for col in asset_cols:
        df_single_row[f"{col}_BIN"] = df_single_row[col].map({"SI": 1, "NO": 0}).astype("Int8")
    df_single_row["INDICE_BIENES"] = df_single_row[[f"{col}_BIN" for col in asset_cols]].sum(axis=1)
    df_single_row.drop(columns=[c for c in asset_cols] + [f"{c}_BIN" for c in asset_cols if f"{c}_BIN" in df_single_row.columns], inplace=True, errors='ignore')


    # Re-apply Ordinal and Label Encoding
    for col_name, encoder_obj in encoders.items():
        if col_name in df_single_row.columns:
            # Ensure the input for transform is 2D for sklearn encoders
            if 'LabelEncoder' in str(type(encoder_obj)) or 'OrdinalEncoder' in str(type(encoder_obj)):
                df_single_row[col_name] = encoder_obj.transform(df_single_row[[col_name]])
            else:
                # Handle cases where encoder_obj might be a raw array (less likely but defensively)
                mapping = {class_name: i for i, class_name in enumerate(encoder_obj)}
                df_single_row[col_name] = df_single_row[col_name].map(mapping).fillna(-1).astype(int) # -1 for unknown

    # Ensure all independent_vars are present and are numeric type
    for col in independent_vars:
        if col not in df_single_row.columns:
            # If a column was dropped or not directly from input, ensure it's added and numeric.
            # For example, numerical inputs from sliders directly provide numeric values.
            df_single_row[col] = pd.to_numeric(None, errors='coerce') # Add with NaN, will be handled by scaler if needed, but should be filled by user_input
        df_single_row[col] = pd.to_numeric(df_single_row[col], errors='coerce')
    
    # Convert any remaining columns to numeric, which were not explicitly encoded or are direct numeric inputs
    for col in df_single_row.columns:
        if col not in encoders and col not in asset_cols: # Don't convert asset_cols, they were used for INDICE_BIENES
            df_single_row[col] = pd.to_numeric(df_single_row[col], errors='coerce')

    # Reorder columns to match the training data's feature order
    # Ensure all columns in independent_vars are in df_single_row before reordering
    for col in independent_vars:
        if col not in df_single_row.columns:
            df_single_row[col] = np.nan # Or a default value if appropriate

    df_single_row = df_single_row[independent_vars].copy()

    # Scale numerical variables using the loaded MinMaxScaler
    df_scaled = pd.DataFrame(scaler.transform(df_single_row), columns=independent_vars)

    return df_scaled

# --- 3. Diseño de la Interfaz de Usuario (UI) de Streamlit ---
st.set_page_config(
    page_title="Predicción Puntajes ICFES Saber 11",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📊 Predicción de Puntajes ICFES Saber 11")
st.markdown("### Aplicación de Machine Learning - Universidad Pontificia Bolivariana")

st.markdown(
    "Esta aplicación predice los puntajes del examen ICFES Saber 11 "
    "basado en variables socioeconómicas, familiares e institucionales. "
    "Ingrese los datos a continuación para obtener las predicciones."
)

st.write("---")
st.subheader("Datos del Estudiante y el Colegio")

user_input = {}

# Group inputs into columns for better layout
col1, col2 = st.columns(2)

with col1:
    user_input['ESTU_GENERO'] = st.radio(
        "Género del Estudiante",
        options=encoders['ESTU_GENERO'].classes_,
        format_func=lambda x: "Femenino" if x == "F" else "Masculino"
    )
    user_input['EDAD'] = st.slider(
        "Edad del Estudiante (años)",
        min_value=12, max_value=60, value=17, step=1
    )
    user_input['COLE_AREA_UBICACION'] = st.selectbox(
        "Área de Ubicación del Colegio",
        options=encoders['COLE_AREA_UBICACION'].classes_
    )
    user_input['COLE_CALENDARIO'] = st.selectbox(
        "Calendario del Colegio",
        options=encoders['COLE_CALENDARIO'].classes_
    )
    user_input['COLE_JORNADA'] = st.selectbox(
        "Jornada del Colegio",
        options=encoders['COLE_JORNADA'].classes_
    )

    # FAMI_CUARTOSHOGAR is numerical (from pipeline R08 and 4.5)
    user_input['FAMI_CUARTOSHOGAR'] = st.slider(
        "Número de Cuartos en el Hogar",
        min_value=1, max_value=20, value=3, step=1 # Based on R08 min/max
    )
    user_input['FAMI_ESTRATOVIVIENDA'] = st.selectbox(
        "Estrato de Vivienda Familiar",
        options=encoders['FAMI_ESTRATOVIVIENDA'].categories_[0]
    )
    # FAMI_PERSONASHOGAR is numerical (from pipeline R08 and 4.5)
    user_input['FAMI_PERSONASHOGAR'] = st.slider(
        "Número de Personas en el Hogar",
        min_value=1, max_value=30, value=4, step=1 # Based on R08 min/max
    )

with col2:
    user_input['ANIO'] = st.slider(
        "Año de Presentación del Examen",
        min_value=2014, max_value=2026, value=2024, step=1
    )
    user_input['TRIMESTRE'] = st.radio(
        "Trimestre de Presentación del Examen",
        options=[1, 2, 3, 4],
        horizontal=True
    )
    user_input['COLE_BILINGUE'] = st.radio(
        "Colegio Bilingüe",
        options=encoders['COLE_BILINGUE'].classes_,
        horizontal=True
    )
    # Asset columns for INDICE_BIENES
    st.markdown("**Tenencia de Bienes en el Hogar**")
    user_input['FAMI_TIENEAUTOMOVIL'] = st.radio(
        "Tiene Automóvil",
        options=["SI", "NO"], key="auto", horizontal=True
    )
    user_input['FAMI_TIENECOMPUTADOR'] = st.radio(
        "Tiene Computador",
        options=["SI", "NO"], key="comp", horizontal=True
    )
    user_input['FAMI_TIENEINTERNET'] = st.radio(
        "Tiene Internet",
        options=["SI", "NO"], key="int", horizontal=True
    )
    user_input['FAMI_TIENELAVADORA'] = st.radio(
        "Tiene Lavadora",
        options=["SI", "NO"], key="lav", horizontal=True
    )

    # Education levels are ordinal, so use options from encoder's categories
    edu_options_madre = encoders['FAMI_EDUCACIONMADRE'].categories_[0]
    user_input['FAMI_EDUCACIONMADRE'] = st.selectbox(
        "Nivel Educativo de la Madre",
        options=edu_options_madre
    )
    edu_options_padre = encoders['FAMI_EDUCACIONPADRE'].categories_[0]
    user_input['FAMI_EDUCACIONPADRE'] = st.selectbox(
        "Nivel Educativo del Padre",
        options=edu_options_padre
    )


# Prediction Button
st.write("---")
if st.button("Predecir Puntajes", type="primary"):
    if not loaded_models: # Check if models were loaded successfully
        st.error("No se pudieron cargar los modelos de ML. Verifique la existencia de los archivos .joblib.")
    else:
        with st.spinner('Realizando predicciones...'):
            # Preprocess user input
            processed_input = preprocess_input(user_input)

            # Generate predictions for each target
            st.subheader("Resultados de la Predicción:")
            predictions = {}
            for target, model in loaded_models.items():
                pred = model.predict(processed_input)[0]
                predictions[target] = pred

            # Display predictions
            results_df = pd.DataFrame({
                'Puntaje': list(predictions.keys()),
                'Valor Predicho': [f"{v:.2f}" for v in predictions.values()]
            })
            st.dataframe(results_df.set_index('Puntaje'), use_container_width=True)

st.write("---")
st.caption("Desarrollado por Sebastian Muñoz, Ivan Velasco y Sebastian Velasco para el curso de Aprendizaje de Máquinas - UPB")

# --- Instructions to run the Streamlit app ---
st.markdown("""
## Cómo Ejecutar esta Aplicación Streamlit

1.  **Guarda el código:** Copia todo el código de esta celda y guárdalo en un archivo llamado `app.py`.
2.  **Asegura los artefactos:** Asegúrate de que todos los archivos `.joblib` mencionados (`pipeline_artefacts.joblib`, `icfes_mejor_punt_global.joblib`, etc.) estén en la misma carpeta que `app.py`.
3.  **Abre una terminal:** Navega hasta la carpeta donde guardaste `app.py` y los archivos `.joblib`.
4.  **Ejecuta la aplicación:** En la terminal, ejecuta el siguiente comando:
    ```bash
    streamlit run app.py
    ```
5.  **Accede a la aplicación:** Streamlit abrirá automáticamente la aplicación en tu navegador web. Si no lo hace, te proporcionará una URL para acceder.

### Resumen de Funcionalidad

Esta aplicación web interactiva permite a los usuarios predecir los puntajes del examen ICFES Saber 11 para un estudiante hipotético. Los usuarios ingresan datos sobre el estudiante, su entorno familiar y el colegio a través de una interfaz amigable. Utiliza un pipeline de Machine Learning previamente entrenado, incluyendo:

*   **Carga de Artefactos:** Recupera el `MinMaxScaler`, `LabelEncoder`, `OrdinalEncoder` y los modelos predictivos guardados.
*   **Preprocesamiento:** Transforma la entrada del usuario aplicando las mismas lógicas de ingeniería de características (como la creación de `INDICE_BIENES`) y escalado que se usaron durante el entrenamiento del modelo.
*   **Predicción:** Emplea los modelos de regresión más performantes (identificados durante la fase de evaluación) para cada uno de los seis puntajes del ICFES Saber 11 (`PUNT_GLOBAL`, `PUNT_MATEMATICAS`, `PUNT_INGLES`, `PUNT_LECTURA_CRITICA`, `PUNT_C_NATURALES`, `PUNT_SOCIALES_CIUDADANAS`).
*   **Visualización de Resultados:** Muestra las predicciones de puntaje de manera clara y concisa en la interfaz.
"""
)
