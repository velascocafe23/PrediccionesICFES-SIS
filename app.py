import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# CRÍTICO: set_page_config DEBE ser la primera llamada a Streamlit.
# En la versión original estaba en la línea 84 → crash garantizado.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción ICFES Saber 11",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

h1  { font-family:'IBM Plex Mono',monospace; font-size:1.85rem;
      color:#0f2c5e; letter-spacing:-1px; margin-bottom:0; }
h3  { font-family:'IBM Plex Mono',monospace; color:#0f2c5e; }

.subtitle   { color:#5a6a8a; font-size:0.92rem; margin:0.2rem 0 1.5rem; }
.warn-box   { background:#fff8e1; border-left:4px solid #f59e0b; border-radius:4px;
              padding:0.75rem 1rem; font-size:0.84rem; color:#78450a; margin-bottom:1rem; }

.section-card  { background:#f7f9fc; border-left:4px solid #1a56db;
                 border-radius:4px; padding:1rem 1.2rem; margin-bottom:1rem; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:600;
                 color:#1a56db; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.7rem; }

.result-card  { background:linear-gradient(135deg,#0f2c5e 0%,#1a56db 100%);
                border-radius:8px; padding:1.4rem 1.8rem; color:white; margin-bottom:0.8rem; }
.result-title { font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                letter-spacing:2px; text-transform:uppercase; opacity:0.7; margin-bottom:0.25rem; }
.result-value { font-family:'IBM Plex Mono',monospace; font-size:2.1rem;
                font-weight:600; line-height:1; }
.result-meta  { font-size:0.78rem; opacity:0.72; margin-top:0.3rem; }

.stButton>button { background:#0f2c5e; color:white; border:none; border-radius:4px;
                   padding:0.65rem 2.5rem; font-family:'IBM Plex Mono',monospace;
                   font-size:0.85rem; font-weight:600; letter-spacing:0.5px; width:100%; }
.stButton>button:hover { background:#1a56db; }

.footer { margin-top:3rem; padding-top:1rem; border-top:1px solid #e2e8f0;
          font-size:0.76rem; color:#94a3b8; text-align:center;
          font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE ARTEFACTOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    """
    Carga pipeline_artefacts.joblib y los 6 modelos individuales.
    @st.cache_resource garantiza que se ejecuta una sola vez.
    """
    path = "pipeline_artefacts.joblib"
    if not os.path.exists(path):
        st.error(f"No se encontró '{path}'. Coloca todos los .joblib en la misma carpeta que app.py.")
        st.stop()

    arts       = joblib.load(path)
    scaler     = arts["scaler"]            # MinMaxScaler fitted
    encoders   = arts["encoders"]          # {col: OrdinalEncoder | LabelEncoder}
    indep_vars = arts["independent_vars"]  # lista ordenada igual que al entrenar
    score_cols = arts["score_cols"]        # 6 PUNT_*
    asset_cols = arts["asset_cols"]        # FAMI_TIENE*

    # Cargar modelos individuales.
    # Cada .joblib contiene: modelo_fit, vars_sig, metricas, model_name
    loaded_models = {}
    for target in score_cols:
        fname = f"icfes_mejor_{target.lower()}.joblib"
        if not os.path.exists(fname):
            st.error(f"Modelo no encontrado: {fname}")
            st.stop()
        md = joblib.load(fname)
        loaded_models[target] = {
            "modelo" : md["modelo_fit"],
            "vars"   : md["vars_sig"],   # subconjunto reducido por significancia estadística
            "nombre" : md.get("model_name", target),
            "r2"     : md.get("metricas", {}).get("r2_test"),
        }

    return scaler, encoders, indep_vars, score_cols, asset_cols, loaded_models


scaler, encoders, indep_vars, score_cols, asset_cols, loaded_models = load_artefacts()


# ── Helper: clases de cualquier tipo de encoder ───────────────────────────────
def get_clases(enc):
    """Devuelve lista de categorías para OrdinalEncoder o LabelEncoder."""
    if enc is None:
        return []
    if hasattr(enc, "categories_"):   # OrdinalEncoder
        return list(enc.categories_[0])
    if hasattr(enc, "classes_"):      # LabelEncoder
        return list(enc.classes_)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO
#
# Replica EXACTA del pipeline de entrenamiento en el mismo orden:
#   §7.1  → INDICE_BIENES
#   §8.2  → OrdinalEncoding  (FAMI_EDUCACION*, FAMI_ESTRATOVIVIENDA)
#   §8.3  → LabelEncoding    (ESTU_GENERO, COLE_*, COLE_BILINGUE...)
#   §8.5  → Interacciones económicas + log transforms  ← VARIABLES NUEVAS
#   §7.4  → Variables contextuales (imputadas en producción)
#   §9    → MinMaxScaler
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # ── §7.1  INDICE_BIENES ──────────────────────────────────────────────────
    bin_cols = []
    for col in asset_cols:
        if col in df.columns:
            df[f"{col}_BIN"] = df[col].map({"SI": 1, "NO": 0}).fillna(0).astype(int)
            bin_cols.append(f"{col}_BIN")
    df["INDICE_BIENES"] = df[bin_cols].sum(axis=1).astype(float)
    # Eliminar columnas de bienes originales (no van al modelo)
    df.drop(columns=[c for c in asset_cols if c in df.columns], inplace=True, errors="ignore")
    df.drop(columns=bin_cols, inplace=True, errors="ignore")

    # ── §8.2 + §8.3  ENCODING ───────────────────────────────────────────────
    # CRÍTICO: OrdinalEncoder.transform() espera shape (n, 1)  → df[[col]]
    #          LabelEncoder.transform()  espera shape (n,)     → df[col]
    # La versión original usaba df[[col]] para ambos → ValueError en LabelEncoder
    for col, enc in encoders.items():
        if col not in df.columns:
            continue
        enc_tipo = type(enc).__name__
        try:
            if enc_tipo == "OrdinalEncoder":
                df[col] = enc.transform(df[[col]]).flatten()
            elif enc_tipo == "LabelEncoder":
                df[col] = enc.transform(df[col].astype(str))
            else:
                mapping = {cls: i for i, cls in enumerate(get_clases(enc))}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        except Exception:
            df[col] = -1   # categoría no vista en entrenamiento → valor neutro

    # ── §8.5  INTERACCIONES ECONÓMICAS + LOG TRANSFORMS ─────────────────────
    # La versión original NO calculaba estas variables.
    # Si faltan, el scaler recibe NaN en esas posiciones y las predicciones
    # quedan desplazadas respecto al entrenamiento.
    def fe(nombre, fn):
        try:
            val = fn(df).replace([np.inf, -np.inf], np.nan)
            df[nombre] = val
        except Exception:
            df[nombre] = np.nan

    fe("ESTRATO_X_EDU_MADRE",
       lambda d: d["FAMI_ESTRATOVIVIENDA"] * d["FAMI_EDUCACIONMADRE"])

    fe("ESTRATO_X_EDU_PADRE",
       lambda d: d["FAMI_ESTRATOVIVIENDA"] * d["FAMI_EDUCACIONPADRE"])

    fe("DENSIDAD_HOGAR",
       lambda d: d["FAMI_PERSONASHOGAR"] / (d["FAMI_CUARTOSHOGAR"] + 1))

    fe("INTERNET_X_EDU_MADRE",
       lambda d: d["FAMI_TIENEINTERNET"] * d["FAMI_EDUCACIONMADRE"]
                 if "FAMI_TIENEINTERNET" in d.columns
                 else d["FAMI_EDUCACIONMADRE"] * 0)

    fe("LOG_PERSONAS",
       lambda d: np.log1p(d["FAMI_PERSONASHOGAR"].clip(lower=0)))

    fe("LOG_CUARTOS",
       lambda d: np.log1p(d["FAMI_CUARTOSHOGAR"].clip(lower=0)))

    # ── §7.4  VARIABLES CONTEXTUALES (no replicables en producción) ──────────
    # PROM_GLOBAL_MCPIO y PROM_GLOBAL_COLEGIO se calcularon con groupby
    # sobre el dataset completo. En producción se imputa con 0.5
    # (valor medio dentro del rango [0,1] post-escalado).
    for col in ["PROM_GLOBAL_MCPIO", "PROM_GLOBAL_COLEGIO"]:
        if col in indep_vars and col not in df.columns:
            df[col] = 0.5

    # ── Garantizar todas las columnas esperadas y tipos numéricos ────────────
    for col in indep_vars:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── §9  MinMaxScaler ─────────────────────────────────────────────────────
    df = df[indep_vars].copy()   # orden idéntico al del entrenamiento
    df_scaled = pd.DataFrame(scaler.transform(df), columns=indep_vars)

    return df_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 3. UI — CABECERA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🎓 Predicción de Puntajes ICFES Saber 11</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Modelo de Machine Learning &nbsp;·&nbsp; '
    'Universidad Pontificia Bolivariana &nbsp;·&nbsp; 2026</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="warn-box">⚠️ <strong>Nota:</strong> Las predicciones son estimaciones '
    'estadísticas basadas en datos históricos. No constituyen garantía de resultados reales.</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORMULARIO DE ENTRADA  (3 columnas)
# ─────────────────────────────────────────────────────────────────────────────
col_est, col_cole, col_fam = st.columns(3)

# ── Columna A: Estudiante ─────────────────────────────────────────────────────
with col_est:
    st.markdown('<div class="section-card"><div class="section-title">👤 Estudiante</div>',
                unsafe_allow_html=True)

    genero_opts = get_clases(encoders.get("ESTU_GENERO")) or ["F", "M"]
    genero = st.radio(
        "Género",
        options=genero_opts,
        format_func=lambda x: "Femenino" if x == "F" else "Masculino",
        horizontal=True,
    )
    edad      = st.slider("Edad (años)", min_value=12, max_value=30, value=17)
    anio      = st.slider("Año del examen", min_value=2014, max_value=2026, value=2024)
    trimestre = st.radio("Trimestre", options=[1, 2, 3, 4], horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Columna B: Colegio ────────────────────────────────────────────────────────
with col_cole:
    st.markdown('<div class="section-card"><div class="section-title">🏫 Colegio</div>',
                unsafe_allow_html=True)

    area = st.selectbox(
        "Área de ubicación",
        get_clases(encoders.get("COLE_AREA_UBICACION")) or ["URBANO", "RURAL"],
    )
    calendario = st.selectbox(
        "Calendario",
        get_clases(encoders.get("COLE_CALENDARIO")) or ["A", "B"],
    )
    jornada = st.selectbox(
        "Jornada",
        get_clases(encoders.get("COLE_JORNADA")) or ["MANANA", "TARDE", "COMPLETA"],
    )
    # COLE_CARACTER faltaba completamente en la versión original
    caracter = st.selectbox(
        "Carácter del colegio",
        get_clases(encoders.get("COLE_CARACTER")) or ["ACADEMICO", "TECNICO", "OTRO"],
    )
    bilingue_opts = get_clases(encoders.get("COLE_BILINGUE")) or ["S", "N"]
    bilingue = st.radio("Bilingüe", options=bilingue_opts, horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Columna C: Familia ────────────────────────────────────────────────────────
with col_fam:
    st.markdown('<div class="section-card"><div class="section-title">🏠 Entorno Familiar</div>',
                unsafe_allow_html=True)

    estrato_opts = get_clases(encoders.get("FAMI_ESTRATOVIVIENDA")) or \
                  ["Sin Estrato","Estrato 1","Estrato 2","Estrato 3",
                   "Estrato 4","Estrato 5","Estrato 6"]
    estrato   = st.selectbox("Estrato de vivienda", options=estrato_opts)

    edu_opts  = get_clases(encoders.get("FAMI_EDUCACIONMADRE")) or ["Ninguno", "Postgrado"]
    edu_madre = st.selectbox("Educación de la madre", options=edu_opts)
    edu_padre = st.selectbox(
        "Educación del padre",
        options=get_clases(encoders.get("FAMI_EDUCACIONPADRE")) or edu_opts,
    )

    personas = st.slider("Personas en el hogar", min_value=1, max_value=15, value=4)
    cuartos  = st.slider("Cuartos en el hogar",  min_value=1, max_value=15, value=3)

    st.markdown("**Bienes del hogar**")
    ca, cb = st.columns(2)
    with ca:
        tiene_auto = st.checkbox("Automóvil")
        tiene_comp = st.checkbox("Computador", value=True)
    with cb:
        tiene_inet = st.checkbox("Internet", value=True)
        tiene_lava = st.checkbox("Lavadora",  value=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOTÓN Y PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.write("")
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predecir = st.button("⚡ Calcular predicciones", type="primary")

if predecir:
    # Construir dict crudo con los mismos nombres de columna que el pipeline
    user_raw = {
        "ESTU_GENERO"          : genero,
        "EDAD"                 : float(edad),
        "ANIO"                 : float(anio),
        "TRIMESTRE"            : float(trimestre),
        "COLE_AREA_UBICACION"  : area,
        "COLE_CALENDARIO"      : calendario,
        "COLE_JORNADA"         : jornada,
        "COLE_CARACTER"        : caracter,
        "COLE_BILINGUE"        : bilingue,
        "FAMI_ESTRATOVIVIENDA" : estrato,
        "FAMI_EDUCACIONMADRE"  : edu_madre,
        "FAMI_EDUCACIONPADRE"  : edu_padre,
        "FAMI_PERSONASHOGAR"   : float(personas),
        "FAMI_CUARTOSHOGAR"    : float(cuartos),
        "FAMI_TIENEAUTOMOVIL"  : "SI" if tiene_auto else "NO",
        "FAMI_TIENECOMPUTADOR" : "SI" if tiene_comp else "NO",
        "FAMI_TIENEINTERNET"   : "SI" if tiene_inet else "NO",
        "FAMI_TIENELAVADORA"   : "SI" if tiene_lava else "NO",
        # COLE_MCPIO_UBICACION no disponible en producción
        # → imputado como promedio global en preprocess_input()
    }

    with st.spinner("Calculando predicciones…"):
        try:
            X_scaled = preprocess_input(user_raw)
        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("### Resultados de la predicción")

    nombres = {
        "PUNT_GLOBAL"              : ("Puntaje Global",        "📊"),
        "PUNT_MATEMATICAS"         : ("Matemáticas",           "🔢"),
        "PUNT_INGLES"              : ("Inglés",                "🇬🇧"),
        "PUNT_LECTURA_CRITICA"     : ("Lectura Crítica",       "📖"),
        "PUNT_C_NATURALES"         : ("Ciencias Naturales",    "🔬"),
        "PUNT_SOCIALES_CIUDADANAS" : ("Sociales y Ciudadanas", "🏛️"),
    }

    cols_res = st.columns(3)
    tabla    = []

    for i, (target, info) in enumerate(loaded_models.items()):
        modelo   = info["modelo"]
        vars_sig = info["vars"]   # columnas con las que se entrenó el modelo

        # CRÍTICO: cada modelo fue entrenado con un subconjunto vars_sig,
        # no con todo indep_vars. Pasarle columnas extra → shape mismatch.
        cols_ok = [v for v in vars_sig if v in X_scaled.columns]
        X_target = X_scaled[cols_ok]

        try:
            pred = float(modelo.predict(X_target)[0])
            pred = max(0.0, pred)   # puntajes no pueden ser negativos
        except Exception as e:
            pred = np.nan
            st.warning(f"Error prediciendo {target}: {e}")

        label, icon = nombres.get(target, (target, "📌"))
        r2_txt = f"R² = {info['r2']:.3f}" if info["r2"] is not None else ""

        with cols_res[i % 3]:
            valor_str = f"{pred:.1f}" if not np.isnan(pred) else "—"
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">{icon} {label}</div>
                <div class="result-value">{valor_str}</div>
                <div class="result-meta">{info['nombre']} &nbsp;|&nbsp; {r2_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        tabla.append({
            "Prueba"           : label,
            "Puntaje predicho" : valor_str,
            "Modelo"           : info["nombre"],
            "R² test"          : f"{info['r2']:.4f}" if info["r2"] else "—",
            "Variables usadas" : len(vars_sig),
        })

    st.markdown("")
    st.dataframe(
        pd.DataFrame(tabla).set_index("Prueba"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Sebastián Muñoz &nbsp;·&nbsp; Iván Velasco &nbsp;·&nbsp; Sebastián Velasco
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Aprendizaje de Máquinas &nbsp;·&nbsp; Universidad Pontificia Bolivariana &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
