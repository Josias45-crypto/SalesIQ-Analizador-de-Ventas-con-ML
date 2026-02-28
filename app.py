# ============================================================
# SALESIQ PRO — Analizador de Ventas con ML
# ============================================================
# Librerías utilizadas:
#   - streamlit   → interfaz web
#   - pandas      → manipulación de datos (HT-01)
#   - numpy       → cálculos matemáticos (HT-01)
#   - scikit-learn→ modelos de predicción (HT-02)
#   - plotly      → gráficas interactivas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(
    page_title="SalesIQ Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS — Corporativo Premium
# Paleta: Azul marino #0b1e3d · Dorado #c9a84c · Crema #f8f6f1
# Tipografía: Cormorant Garamond (display) + Plus Jakarta Sans (body)
# Animaciones: fadeUp escalonado, shimmer en KPIs, pulse en alertas
# NOTA: NO se oculta el header completo para preservar el botón del sidebar
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── PALETA CORPORATIVA PREMIUM ────────────────────────────
   Charcoal  #1c1f26   fondo sidebar/hero (casi negro azulado)
   Obsidian  #252930   variante oscura
   Champagne #d4b896   acento cálido principal
   Sand      #ede8df   fondo principal (arena cálida)
   Surface   #ffffff   tarjetas
   Stone     #f5f2ed   tarjeta alternativa
   Slate     #5a6270   texto secundario
   Border    #ddd8cf   bordes suaves
   Emerald   #1a6b52   éxito/positivo
   Crimson   #a83232   danger
   Amber     #9a6f10   warning
   Ink       #1c1f26   texto principal
────────────────────────────────────────────────────────── */
:root {
    --navy:    #1c1f26;
    --navy2:   #252930;
    --steel:   #363d4a;
    --gold:    #d4b896;
    --gold2:   #e8d0b0;
    --cream:   #ede8df;
    --surface: #ffffff;
    --border:  #ddd8cf;
    --slate:   #5a6270;
    --muted:   #8a909a;
    --success: #1a6b52;
    --danger:  #a83232;
    --warning: #9a6f10;
    --text:    #1c1f26;
}

/* ── RESET GLOBAL ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}
.stApp { background: var(--cream); }

/* ── OCULTAR CHROME DE STREAMLIT ───────────────────────── */
/* Solo ocultamos lo necesario — NO ocultamos header completo
   para que el botón › del sidebar siga funcionando */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── SIDEBAR ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(201,168,76,0.15) !important;
}
[data-testid="stSidebar"] * {
    color: #a8b8cc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e8edf2 !important; }

/* Slider track en dorado */
[data-testid="stSlider"] > div > div > div {
    background: var(--gold) !important;
}

/* ── BOTÓN PRINCIPAL ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold2) 100%) !important;
    color: #1a0a00 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(201,168,76,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(201,168,76,0.40) !important;
}

/* ── BOTÓN DESCARGA ────────────────────────────────────── */
.stDownloadButton > button {
    background: var(--navy) !important;
    color: var(--gold) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 12px !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: var(--navy2) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 4px 16px rgba(201,168,76,0.2) !important;
}

/* ── ANIMACIONES DE ENTRADA ────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
/* Shimmer sobre tarjetas KPI */
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
/* Pulso suave para alertas críticas */
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(192,57,43,0); }
    50%       { box-shadow: 0 0 0 4px rgba(192,57,43,0.15); }
}
/* Línea dorada deslizante en el hero */
@keyframes slideRight {
    from { width: 0; }
    to   { width: 80px; }
}
/* Contador numérico */
@keyframes countUp {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}

.anim-1 { animation: fadeUp 0.5s ease 0.05s both; }
.anim-2 { animation: fadeUp 0.5s ease 0.12s both; }
.anim-3 { animation: fadeUp 0.5s ease 0.19s both; }
.anim-4 { animation: fadeUp 0.5s ease 0.26s both; }
.anim-5 { animation: fadeUp 0.5s ease 0.33s both; }
.anim-6 { animation: fadeUp 0.5s ease 0.40s both; }

/* ── HERO ──────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy2) 60%, var(--steel) 100%);
    border-radius: 20px;
    padding: 52px 56px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.6s ease both;
}
/* Textura de puntos decorativos */
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(circle, rgba(201,168,76,0.08) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
}
/* Orbe dorado en la esquina */
.hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(201,168,76,0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    animation: fadeUp 0.5s ease 0.1s both;
}
.hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.15);
    color: var(--gold2);
    border: 1px solid rgba(201,168,76,0.3);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 10px;
    letter-spacing: 3px;
    font-weight: 600;
    text-transform: uppercase;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, var(--gold), transparent);
    animation: slideRight 0.8s ease 0.4s both;
    width: 0;
}
.hero h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 4.5vw, 3.6rem);
    font-weight: 600;
    color: white;
    margin: 0 0 12px;
    line-height: 1.1;
    letter-spacing: -0.5px;
    animation: fadeUp 0.5s ease 0.2s both;
}
.hero h1 span {
    color: var(--gold2);
    font-style: italic;
}
.hero p {
    color: #7a90a8;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.7;
    animation: fadeUp 0.5s ease 0.3s both;
}
/* Stats en el hero (esquina derecha) */
.hero-stats {
    position: absolute;
    right: 56px;
    top: 50%;
    transform: translateY(-50%);
    text-align: right;
    animation: fadeIn 0.8s ease 0.5s both;
}
.hero-stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    color: var(--gold);
    line-height: 1;
    font-weight: 600;
    letter-spacing: -2px;
}
.hero-stat-lbl {
    font-size: 10px;
    letter-spacing: 3px;
    color: #4a6080;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── SEPARADOR DE SECCIÓN ──────────────────────────────── */
.sec-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--navy);
    margin: 40px 0 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    position: relative;
}
/* Línea dorada bajo el título */
.sec-title::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 48px; height: 2px;
    background: var(--gold);
    border-radius: 2px;
}

/* ── KPI CARD ──────────────────────────────────────────── */
.kpi-card {
    background: var(--surface);
    border-radius: 16px;
    padding: 26px 24px 22px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(11,30,61,0.06), 0 4px 16px rgba(11,30,61,0.04);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
}
/* Shimmer al hover */
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(201,168,76,0.06) 50%,
        transparent 100%);
    transition: left 0.5s ease;
}
.kpi-card:hover::before { left: 150%; }
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(11,30,61,0.12), 0 2px 8px rgba(11,30,61,0.08);
}
/* Acento de color en la parte superior de la card */
.kpi-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 12px;
    margin-top: 6px;
}
.kpi-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--navy);
    line-height: 1;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
    animation: countUp 0.5s ease both;
}
.kpi-sub {
    font-size: 12px;
    color: var(--slate);
    font-weight: 400;
    line-height: 1.4;
}
.kpi-bar { height: 2px; border-radius: 2px; margin-top: 18px; opacity: 0.7; }

/* ── DISCLAIMER ────────────────────────────────────────── */
.disclaimer {
    background: linear-gradient(135deg, #fffdf5 0%, #fff9e8 100%);
    border: 1px solid rgba(184,134,11,0.2);
    border-left: 4px solid var(--warning);
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    font-size: 12px;
    color: #7a5a0a;
    line-height: 1.7;
    margin: 16px 0 24px;
}

/* ── TARJETA PREDICCIÓN ────────────────────────────────── */
.pred-card {
    background: linear-gradient(160deg, var(--navy) 0%, var(--navy2) 50%, #1e3a5f 100%);
    border-radius: 18px;
    padding: 32px 28px;
    text-align: center;
    border: 1px solid rgba(201,168,76,0.2);
    box-shadow: 0 8px 32px rgba(11,30,61,0.3);
    position: relative;
    overflow: hidden;
    height: 100%;
}
.pred-card::before {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 65%);
    pointer-events: none;
}
/* Puntos decorativos */
.pred-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(circle, rgba(201,168,76,0.05) 1px, transparent 1px);
    background-size: 20px 20px;
    pointer-events: none;
}
.pred-tag {
    font-size: 10px;
    letter-spacing: 3px;
    color: rgba(201,168,76,0.5);
    text-transform: uppercase;
    margin-bottom: 14px;
    font-weight: 600;
}
.pred-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: var(--gold2);
    line-height: 1;
    margin-bottom: 8px;
    letter-spacing: -1px;
}
.pred-range { font-size: 12px; color: #4a6080; margin-bottom: 18px; }
.conf-track {
    background: rgba(255,255,255,0.08);
    border-radius: 4px; height: 5px;
    margin: 8px 0; overflow: hidden;
}
.conf-fill {
    height: 5px; border-radius: 4px;
    background: linear-gradient(90deg, var(--gold), var(--gold2));
    transition: width 1s ease;
}
.pred-model {
    font-size: 11px;
    color: rgba(201,168,76,0.35);
    margin-top: 14px;
    letter-spacing: 1px;
}

/* ── PRODUCTO ROW ──────────────────────────────────────── */
.prod-row {
    background: var(--surface);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 6px;
    border: 1px solid var(--border);
    display: grid;
    grid-template-columns: 38px 1fr auto;
    gap: 14px;
    align-items: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
}
.prod-row:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(11,30,61,0.08);
    border-color: rgba(201,168,76,0.3);
}
.rank-dot {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── ALERTA ITEM ───────────────────────────────────────── */
.alert-item {
    background: var(--surface);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid;
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    transition: transform 0.2s ease;
}
.alert-item:hover { transform: translateX(3px); }
.alert-critico { animation: pulse-red 2.5s ease infinite; }

/* ── BIENVENIDA ────────────────────────────────────────── */
.welcome-card {
    background: var(--surface);
    border-radius: 20px;
    padding: 64px 48px;
    text-align: center;
    border: 1px solid var(--border);
    box-shadow: 0 4px 24px rgba(11,30,61,0.06);
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(201,168,76,0.06) 0%, transparent 65%);
    pointer-events: none;
}

/* ── SIDEBAR LOGO ──────────────────────────────────────── */
.sb-logo {
    text-align: center;
    padding: 32px 0 24px;
    border-bottom: 1px solid rgba(201,168,76,0.12);
    margin-bottom: 24px;
}
.sb-logo-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--gold), var(--gold2));
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(201,168,76,0.3);
}
.sb-logo-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: white !important;
    letter-spacing: -0.3px;
}
.sb-logo-tag {
    font-size: 9px;
    letter-spacing: 4px;
    color: rgba(201,168,76,0.5) !important;
    text-transform: uppercase;
    margin-top: 3px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES — Sin cambios (lógica intacta)
# ============================================================

def cargar_datos(archivo):
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo, encoding='utf-8', encoding_errors='replace')
        else:
            df = pd.read_excel(archivo)
        df.columns = df.columns.str.strip().str.lower()
        mapeo = {
            'producto':  ['producto','product','item','articulo','artículo','descripcion','descripción','nombre_producto'],
            'fecha':     ['fecha','date','mes','periodo','período','fecha_venta','date_sale'],
            'ventas':    ['ventas','sales','total','monto','ingresos','revenue','importe','valor','precio_total'],
            'cantidad':  ['cantidad','qty','units','unidades','cant','cantidad_vendida','piezas'],
            'costo':     ['costo','cost','precio_costo','cog','coste','costo_unitario'],
            'categoria': ['categoria','categoría','category','tipo','type','grupo','línea','linea'],
            'precio':    ['precio','price','precio_unitario','unit_price','pvp'],
        }
        col_map = {}
        for estandar, variantes in mapeo.items():
            for col in df.columns:
                if col in variantes and estandar not in col_map.values():
                    col_map[col] = estandar
                    break
        df = df.rename(columns=col_map)
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)
            df = df.dropna(subset=['fecha'])
            df['mes']        = df['fecha'].dt.to_period('M').astype(str)
            df['mes_num']    = df['fecha'].dt.month
            df['anio']       = df['fecha'].dt.year
            df['trimestre']  = df['fecha'].dt.quarter
            df['dia_semana'] = df['fecha'].dt.dayofweek
        for col in ['ventas', 'cantidad', 'costo', 'precio']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'ventas' in df.columns and 'costo' in df.columns:
            df['ganancia']   = df['ventas'] - df['costo']
            df['margen_pct'] = np.where(df['ventas'] > 0, (df['ganancia'] / df['ventas']) * 100, 0)
        if 'ventas' in df.columns:
            df = df[df['ventas'] > 0].reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)


def calcular_kpis(df):
    kpis = {}
    if 'ventas' in df.columns:
        v = df['ventas'].values
        kpis['total_ventas']   = np.sum(v)
        kpis['venta_promedio'] = np.mean(v)
        kpis['venta_mediana']  = np.median(v)
        kpis['venta_max']      = np.max(v)
        kpis['variabilidad']   = np.std(v)
        if 'mes' in df.columns:
            ventas_x_mes = df.groupby('mes')['ventas'].sum().values
            kpis['promedio_mensual'] = np.mean(ventas_x_mes)
            kpis['mejor_mes_valor']  = np.max(ventas_x_mes)
            kpis['peor_mes_valor']   = np.min(ventas_x_mes)
            kpis['n_meses']          = len(ventas_x_mes)
    if 'cantidad' in df.columns:
        kpis['total_unidades'] = int(np.sum(df['cantidad'].values))
    if 'ganancia' in df.columns:
        g = df['ganancia'].values
        kpis['total_ganancia']  = np.sum(g)
        kpis['margen_promedio'] = np.mean(df['margen_pct'].values)
    if 'producto' in df.columns:
        kpis['n_productos'] = df['producto'].nunique()
    return kpis


def predecir_ventas(df):
    if 'mes' not in df.columns or 'ventas' not in df.columns:
        return None, None, None, "Sin columna fecha"
    vm = df.groupby('mes')['ventas'].sum().reset_index().sort_values('mes')
    if len(vm) < 4:
        return None, None, None, "Se necesitan mínimo 4 meses de datos"
    n = len(vm)
    tiempo    = np.arange(n)
    meses_cal = np.array([int(m.split('-')[1]) for m in vm['mes']])
    trimestres = np.ceil(meses_cal / 3).astype(int)
    X = np.column_stack([tiempo, meses_cal, trimestres])
    y = vm['ventas'].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelos  = {
        'Regresión Lineal':  LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    mejor_modelo = None
    mejor_nombre = ''
    mejor_error  = float('inf')
    resultados   = {}
    for nombre, modelo in modelos.items():
        try:
            cv_scores   = cross_val_score(modelo, X_scaled, y, cv=min(3, n-1),
                                          scoring='neg_mean_absolute_percentage_error')
            error_cv    = -np.mean(cv_scores)
            modelo.fit(X_scaled, y)
            pred_train  = modelo.predict(X_scaled)
            error_train = mean_absolute_percentage_error(y, pred_train)
            error_total = (error_cv + error_train) / 2
            resultados[nombre] = error_total
            if error_total < mejor_error:
                mejor_error  = error_total
                mejor_modelo = modelo
                mejor_nombre = nombre
        except:
            continue
    if mejor_modelo is None:
        return None, None, None, "Error al entrenar modelos"
    siguiente_idx  = n
    siguiente_mes  = (meses_cal[-1] % 12) + 1
    siguiente_trim = int(np.ceil(siguiente_mes / 3))
    X_pred     = scaler.transform([[siguiente_idx, siguiente_mes, siguiente_trim]])
    prediccion = max(0, mejor_modelo.predict(X_pred)[0])
    pred_historial = mejor_modelo.predict(X_scaled)
    errores  = np.abs(y - pred_historial)
    margen   = np.mean(errores) * 1.5
    rango_bajo  = max(0, prediccion - margen)
    rango_alto  = prediccion + margen
    confianza   = max(0, min(95, (1 - mejor_error) * 100))
    return prediccion, (rango_bajo, rango_alto), confianza, mejor_nombre


def analisis_productos(df, n=10):
    if 'producto' not in df.columns or 'ventas' not in df.columns:
        return None
    cols_agg = {'ventas': ['sum', 'mean', 'count']}
    if 'cantidad'   in df.columns: cols_agg['cantidad']   = 'sum'
    if 'ganancia'   in df.columns: cols_agg['ganancia']   = 'sum'
    if 'margen_pct' in df.columns: cols_agg['margen_pct'] = 'mean'
    top = df.groupby('producto').agg(cols_agg)
    top.columns = ['_'.join(c).strip('_') for c in top.columns]
    top = top.rename(columns={
        'ventas_sum': 'ventas', 'ventas_mean': 'ticket_prom',
        'ventas_count': 'transacciones', 'cantidad_sum': 'unidades',
        'ganancia_sum': 'ganancia', 'margen_pct_mean': 'margen'
    })
    top = top.sort_values('ventas', ascending=False).head(n).reset_index()
    return top


def detectar_alertas(df, umbral_stock=10, umbral_caida=0.20):
    alertas = {'critico': [], 'advertencia': [], 'positivo': []}
    if 'producto' not in df.columns:
        return alertas
    if 'mes' in df.columns and 'cantidad' in df.columns:
        ultimo_mes = df['mes'].max()
        reciente   = df[df['mes'] == ultimo_mes]
        por_prod   = reciente.groupby('producto')['cantidad'].sum()
        bajo       = por_prod[por_prod < umbral_stock]
        for prod, cant in bajo.items():
            nivel = 'critico' if cant < umbral_stock // 2 else 'advertencia'
            alertas[nivel].append(f"{prod}: solo {cant:.0f} unidades este mes")
    if 'mes' in df.columns and 'ventas' in df.columns:
        ventas_mes = df.groupby(['producto', 'mes'])['ventas'].sum().unstack(fill_value=0)
        if ventas_mes.shape[1] >= 2:
            ultimo   = ventas_mes.iloc[:, -1]
            anterior = ventas_mes.iloc[:, -2]
            cambio   = np.where(anterior > 0, (ultimo - anterior) / anterior, 0)
            for i, prod in enumerate(ventas_mes.index):
                if cambio[i] < -umbral_caida:
                    alertas['critico'].append(f"{prod}: cayó {abs(cambio[i])*100:.0f}% vs mes anterior")
                elif cambio[i] > 0.20:
                    alertas['positivo'].append(f"{prod}: subió {cambio[i]*100:.0f}% vs mes anterior")
    return alertas


@st.cache_data
def datos_ejemplo():
    np.random.seed(42)
    productos = [
        ('Camisa Slim Fit','Tops',18,45), ('Jeans Clásico','Pantalones',25,65),
        ('Vestido Floral','Vestidos',22,58), ('Chaqueta Cuero','Outerwear',55,120),
        ('Zapatos Oxford','Calzado',35,89), ('Bolso Cuero','Accesorios',45,110),
        ('Gorra Casual','Accesorios',8,22), ('Sudadera Hoodie','Tops',20,55),
        ('Pantalón Chino','Pantalones',22,55), ('Blusa Elegante','Tops',16,42),
    ]
    filas = []
    for mes_num in range(1, 13):
        fecha_base = pd.Timestamp(f'2024-{mes_num:02d}-01')
        for prod, cat, costo, precio in productos:
            tendencia  = 1 + mes_num * 0.025
            estacional = 1.4 if mes_num in [6,7,12] else 0.80 if mes_num in [1,2] else 1.0
            cantidad   = max(1, int(np.random.poisson(18 * tendencia * estacional)))
            filas.append({
                'Fecha':    fecha_base + pd.Timedelta(days=int(np.random.randint(0,27))),
                'Producto': prod, 'Categoria': cat,
                'Cantidad': cantidad,
                'Ventas':   cantidad * precio,
                'Costo':    cantidad * costo,
            })
    return pd.DataFrame(filas)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">📊</div>
        <div class="sb-logo-name">SalesIQ Pro</div>
        <div class="sb-logo-tag">Analytics Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:10px;letter-spacing:2.5px;color:#3a5070;font-weight:700;margin-bottom:10px;'>CARGAR DATOS</p>", unsafe_allow_html=True)
    archivo = st.file_uploader(
        "Excel o CSV de ventas",
        type=['xlsx', 'xls', 'csv'],
        help="Columnas recomendadas: Fecha, Producto, Ventas, Cantidad, Costo, Categoria"
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    usar_ejemplo = st.button("▶ Probar con datos de ejemplo")

    st.markdown("<hr style='border-color:rgba(201,168,76,0.1);margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:10px;letter-spacing:2.5px;color:#3a5070;font-weight:700;margin-bottom:10px;'>CONFIGURACIÓN</p>", unsafe_allow_html=True)
    top_n        = st.slider("Top productos", 3, 20, 10)
    umbral_stock = st.slider("Alerta stock (unidades)", 1, 50, 10)
    umbral_caida = st.slider("Alerta caída de ventas (%)", 5, 50, 20) / 100

    st.markdown("<hr style='border-color:rgba(201,168,76,0.1);margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px;color:#2a4060;line-height:2.1;'>
        <strong style='color:#4a6080;font-size:9px;letter-spacing:2px;'>COLUMNAS ACEPTADAS</strong><br>
        📅 Fecha / Date / Mes<br>
        📦 Producto / Item / Articulo<br>
        💰 Ventas / Sales / Total<br>
        🔢 Cantidad / Qty / Units<br>
        💸 Costo / Cost <em style='color:#1e3050;'>(opcional)</em><br>
        🏷️ Categoria / Category <em style='color:#1e3050;'>(opcional)</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(201,168,76,0.1);margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:10px;color:#2a4060;background:rgba(201,168,76,0.05);
                border-radius:8px;padding:14px;border-left:2px solid rgba(201,168,76,0.3);
                line-height:1.7;'>
        <strong style='color:rgba(201,168,76,0.7);'>⚠️ AVISO LEGAL</strong><br>
        Las predicciones son estimaciones estadísticas basadas en datos históricos.
        No constituyen garantía de resultados futuros.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CARGA DE DATOS
# ============================================================
df = None

if archivo:
    df, error = cargar_datos(archivo)
    if error:
        st.error(f"❌ Error al leer el archivo: {error}")
        df = None
elif usar_ejemplo:
    df_raw = datos_ejemplo()
    import io
    buf = io.BytesIO()
    df_raw.to_excel(buf, index=False)
    buf.seek(0)
    buf.name = 'ejemplo.xlsx'
    df, _ = cargar_datos(buf)
    if df is None:
        df = df_raw.copy()
        df.columns = df.columns.str.lower()
        df['fecha']    = pd.to_datetime(df['fecha'])
        df['mes']      = df['fecha'].dt.to_period('M').astype(str)
        df['ventas']   = df['ventas'].astype(float)
        df['cantidad'] = df['cantidad'].astype(float)
        df['costo']    = df['costo'].astype(float)
        df['ganancia']   = df['ventas'] - df['costo']
        df['margen_pct'] = (df['ganancia'] / df['ventas']) * 100


# ============================================================
# HERO
# ============================================================
n_meses = df['mes'].nunique() if df is not None and 'mes' in df.columns else None

st.markdown(f"""
<div class="hero anim-1">
    <div class="hero-eyebrow">
        <span class="hero-badge">Powered by ML · Scikit-learn + Pandas</span>
        <div class="hero-line"></div>
    </div>
    <h1>SalesIQ <span>Pro</span></h1>
    <p>Inteligencia de ventas con Machine Learning — transforma datos en decisiones
       que hacen crecer cualquier negocio.</p>
    {'<div class="hero-stats"><div class="hero-stat-num">' + str(n_meses) + '</div><div class="hero-stat-lbl">meses analizados</div></div>' if n_meses else ''}
</div>
""", unsafe_allow_html=True)


# ============================================================
# PANTALLA DE BIENVENIDA
# ============================================================
if df is None:
    st.markdown("""
    <div class="welcome-card anim-2">
        <div style="font-size:3.2rem;margin-bottom:20px;">📂</div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:600;
                    color:#0b1e3d;margin-bottom:12px;">
            Sube el Excel del cliente para comenzar
        </div>
        <p style="color:#64748b;max-width:480px;margin:0 auto 8px;line-height:1.8;font-size:14px;">
            Acepta archivos <strong>Excel (.xlsx)</strong> o <strong>CSV</strong>.
            Sin datos ahora — prueba con los datos de ejemplo desde el menú lateral.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, delay, icon, title, desc in [
        (c1, "anim-3", "🔮", "Predicción ML", "3 modelos compiten; el más preciso predice el próximo mes con intervalo de confianza real."),
        (c2, "anim-4", "📊", "Análisis Profundo", "KPIs, top productos, márgenes, tendencias mensuales y comparativa de categorías."),
        (c3, "anim-5", "⚠️", "Alertas Inteligentes", "Detecta caídas de ventas, bajo stock y productos estrella automáticamente."),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {delay}" style="text-align:center;padding:36px 24px;">
                <div class="kpi-accent" style="background:linear-gradient(90deg,#c9a84c,#e8c96a);"></div>
                <div style="font-size:2rem;margin-bottom:14px;">{icon}</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:600;
                            color:#0b1e3d;margin-bottom:8px;">{title}</div>
                <div style="color:#64748b;font-size:13px;line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# DASHBOARD PRINCIPAL
# ============================================================
else:
    st.markdown("""
    <div class="disclaimer anim-2">
        <strong>⚠️ Aviso importante:</strong> Las predicciones son estimaciones estadísticas
        basadas en datos históricos. No representan ni garantizan resultados futuros.
        Úselas como referencia de apoyo, no como única base para decisiones de negocio.
    </div>
    """, unsafe_allow_html=True)

    kpis                          = calcular_kpis(df)
    pred, rango, confianza, modelo_usado = predecir_ventas(df)
    top                           = analisis_productos(df, n=top_n)
    alertas                       = detectar_alertas(df, umbral_stock, umbral_caida)
    vm = df.groupby('mes')['ventas'].sum().reset_index().sort_values('mes') \
         if 'mes' in df.columns else None

    # ── KPIs
    st.markdown('<div class="sec-title anim-2">📌 Resumen Ejecutivo</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    kpi_items = [
        (k1, "anim-2", "VENTAS TOTALES",
         f"${kpis.get('total_ventas',0):,.0f}",
         f"{kpis.get('n_meses','?')} meses analizados",
         "linear-gradient(90deg,#0b1e3d,#2c4a6e)"),
        (k2, "anim-3", "PROMEDIO MENSUAL",
         f"${kpis.get('promedio_mensual',0):,.0f}",
         "Por mes de operación",
         "linear-gradient(90deg,#c9a84c,#e8c96a)"),
        (k3, "anim-3", "UNIDADES VENDIDAS",
         f"{kpis.get('total_unidades',0):,}",
         "Total de piezas",
         "linear-gradient(90deg,#0f7a5a,#1aab7e)"),
        (k4, "anim-4", "GANANCIA NETA",
         f"${kpis.get('total_ganancia',0):,.0f}" if 'total_ganancia' in kpis else f"{kpis.get('n_productos','?')} productos",
         f"Margen {kpis.get('margen_promedio',0):.1f}%" if 'margen_promedio' in kpis else "Productos únicos",
         "linear-gradient(90deg,#1e6b9e,#3a8fc0)"),
        (k5, "anim-5", "PREDICCIÓN",
         f"${pred:,.0f}" if pred else "—",
         f"Próximo mes · {confianza:.0f}% confianza" if pred else "Necesitas +4 meses",
         "linear-gradient(90deg,#7b4ea6,#a070cc)"),
    ]

    for col, delay, label, valor, sub, gradient in kpi_items:
        with col:
            st.markdown(f"""
            <div class="kpi-card {delay}">
                <div class="kpi-accent" style="background:{gradient};"></div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{valor}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gráfica + Predicción
    st.markdown('<div class="sec-title anim-3">📈 Tendencia de Ventas</div>', unsafe_allow_html=True)
    col_graf, col_pred = st.columns([3, 1])

    with col_graf:
        if vm is not None and len(vm) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=vm['mes'], y=vm['ventas'],
                name='Ventas reales',
                marker=dict(
                    color=vm['ventas'],
                    colorscale=[[0, '#162d52'], [0.5, '#2c4a6e'], [1, '#0b1e3d']],
                    line=dict(width=0),
                ),
                opacity=0.88,
            ))
            fig.add_trace(go.Scatter(
                x=vm['mes'], y=vm['ventas'],
                mode='lines+markers',
                name='Tendencia',
                line=dict(color='#c9a84c', width=2.5),
                marker=dict(size=7, color='white', line=dict(color='#c9a84c', width=2)),
            ))
            if pred and rango:
                ultimo_dt   = pd.Period(vm['mes'].iloc[-1], 'M')
                siguiente_p = (ultimo_dt + 1).strftime('%Y-%m')
                fig.add_trace(go.Scatter(
                    x=[siguiente_p], y=[pred],
                    mode='markers',
                    name=f'Predicción ({modelo_usado})',
                    marker=dict(size=16, color='#c9a84c', symbol='diamond',
                                line=dict(color='white', width=2)),
                    error_y=dict(type='data',
                                 array=[rango[1] - pred],
                                 arrayminus=[pred - rango[0]],
                                 visible=True, color='rgba(201,168,76,0.4)', thickness=2),
                ))
            fig.update_layout(
                plot_bgcolor='#fafaf8', paper_bgcolor='white',
                font_family='Plus Jakarta Sans',
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation='h', y=-0.18, font=dict(size=11, color='#64748b')),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#94a3b8'),
                           linecolor='#e8e3d9'),
                yaxis=dict(showgrid=True, gridcolor='#f0ece4',
                           tickprefix='$', tickfont=dict(size=11, color='#94a3b8'),
                           zeroline=False),
                height=340, bargap=0.28,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_pred:
        if pred and rango:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-tag">🔮 Próximo mes</div>
                <div class="pred-value">${pred:,.0f}</div>
                <div class="pred-range">
                    ${rango[0]:,.0f} — ${rango[1]:,.0f}
                </div>
                <div style="font-size:10px;letter-spacing:2px;color:rgba(201,168,76,0.4);
                            text-transform:uppercase;margin-bottom:6px;">Confianza</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{confianza:.0f}%"></div>
                </div>
                <div style="font-size:12px;color:#c9a84c;margin-top:6px;
                            font-weight:600;">{confianza:.0f}%</div>
                <div class="pred-model">Modelo: {modelo_usado}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pred-card">
                <div class="pred-tag">🔮 Predicción</div>
                <div style="color:#3a5070;font-size:13px;margin-top:16px;line-height:1.8;">
                    Se necesitan mínimo 4 meses con columna Fecha para activar el modelo ML.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Productos + Categorías
    st.markdown('<div class="sec-title anim-4">🏆 Análisis de Productos</div>', unsafe_allow_html=True)
    col_top, col_cat = st.columns([3, 2])

    with col_top:
        if top is not None:
            max_v = top['ventas'].max()
            colores_rank = {
                0: ('#c9a84c', '#3d2800'),
                1: ('#9ea8b3', '#1a2530'),
                2: ('#a0734e', 'white'),
            }
            for i, row in top.iterrows():
                color_bg, color_txt = colores_rank.get(i, ('#edf2f7', '#4a6080'))
                pct = (row['ventas'] / max_v) * 100
                margen_str = f"· margen {row['margen']:.0f}%" if 'margen' in row and pd.notna(row.get('margen')) else ""
                gan_str    = f"<span style='color:#0f7a5a;font-size:12px;font-weight:600;'> +${row['ganancia']:,.0f}</span>" if 'ganancia' in row else ""
                units_str  = f"<span style='color:#94a3b8;font-size:11px;'> · {row['unidades']:.0f} uds.</span>" if 'unidades' in row else ""
                st.markdown(f"""
                <div class="prod-row">
                    <div class="rank-dot" style="background:{color_bg};color:{color_txt};">#{i+1}</div>
                    <div>
                        <div style="font-weight:600;color:#0b1e3d;font-size:14px;">{row['producto']}</div>
                        <div style="background:#f0ece4;border-radius:3px;height:2px;margin:7px 0;">
                            <div style="background:linear-gradient(90deg,#c9a84c,#e8c96a);
                                        height:2px;border-radius:3px;width:{pct:.0f}%;"></div>
                        </div>
                        <div style="font-size:11px;color:#94a3b8;">{margen_str}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'Cormorant Garamond',serif;font-size:1.2rem;
                                    color:#0b1e3d;font-weight:600;">${row['ventas']:,.0f}</div>
                        {gan_str}{units_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_cat:
        if 'categoria' in df.columns:
            vc = df.groupby('categoria')['ventas'].sum().sort_values(ascending=False)
            fig2 = go.Figure(go.Pie(
                labels=vc.index, values=vc.values, hole=0.62,
                marker=dict(
                    colors=['#0b1e3d','#c9a84c','#2c4a6e','#0f7a5a','#1e6b9e','#7b4ea6'],
                    line=dict(color='white', width=2),
                ),
                textfont=dict(family='Plus Jakarta Sans', size=12),
                textinfo='label+percent',
            ))
            fig2.update_layout(
                paper_bgcolor='white', font_family='Plus Jakarta Sans',
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, height=360,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Agrega una columna 'Categoria' para ver la distribución.")

    # ── Alertas
    st.markdown('<div class="sec-title anim-5">⚠️ Alertas del Negocio</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("**🔴 Crítico**")
        if alertas['critico']:
            for a in alertas['critico'][:5]:
                st.markdown(f"""
                <div class="alert-item alert-critico" style="border-left-color:#c0392b;">
                    <span>{a}</span>
                    <span style="color:#c0392b;font-size:10px;font-weight:700;
                                 letter-spacing:1px;">URGENTE</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-item' style='border-left-color:#0f7a5a;color:#0f7a5a;'>✅ Sin alertas críticas</div>", unsafe_allow_html=True)

    with a2:
        st.markdown("**🟡 Advertencias**")
        if alertas['advertencia']:
            for a in alertas['advertencia'][:5]:
                st.markdown(f"""
                <div class="alert-item" style="border-left-color:#b8860b;">
                    <span>{a}</span>
                    <span style="color:#b8860b;font-size:10px;font-weight:700;
                                 letter-spacing:1px;">REVISAR</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-item' style='border-left-color:#0f7a5a;color:#0f7a5a;'>✅ Sin advertencias</div>", unsafe_allow_html=True)

    with a3:
        st.markdown("**🟢 Positivos**")
        if alertas['positivo']:
            for a in alertas['positivo'][:5]:
                st.markdown(f"""
                <div class="alert-item" style="border-left-color:#0f7a5a;">
                    <span style="color:#0b1e3d;font-size:13px;">{a}</span>
                    <span style="color:#0f7a5a;font-size:10px;font-weight:700;
                                 letter-spacing:1px;">↑ BIEN</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-item' style='border-left-color:#94a3b8;color:#94a3b8;'>Sin variaciones positivas detectadas</div>", unsafe_allow_html=True)

    # ── Exportar
    st.markdown('<div class="sec-title anim-6">📁 Exportar Reporte</div>', unsafe_allow_html=True)

    def generar_excel(df, kpis, top, pred, rango, confianza, modelo):
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Datos Procesados', index=False)
            filas_kpi = [
                ['Métrica', 'Valor', 'Nota'],
                ['Total Ventas', f"${kpis.get('total_ventas',0):,.2f}", ''],
                ['Promedio Mensual', f"${kpis.get('promedio_mensual',0):,.2f}", ''],
                ['Total Unidades', f"{kpis.get('total_unidades',0):,}", ''],
                ['Ganancia Total', f"${kpis.get('total_ganancia',0):,.2f}" if 'total_ganancia' in kpis else 'N/A', ''],
                ['Margen Promedio', f"{kpis.get('margen_promedio',0):.1f}%" if 'margen_promedio' in kpis else 'N/A', ''],
                ['', '', ''],
                ['PREDICCIÓN ML', '', ''],
                ['Estimación próximo mes', f"${pred:,.2f}" if pred else 'N/A', ''],
                ['Rango mínimo', f"${rango[0]:,.2f}" if rango else 'N/A', ''],
                ['Rango máximo', f"${rango[1]:,.2f}" if rango else 'N/A', ''],
                ['Confianza del modelo', f"{confianza:.0f}%" if confianza else 'N/A', ''],
                ['Modelo utilizado', modelo if modelo else 'N/A', ''],
                ['', '', ''],
                ['⚠️ AVISO LEGAL', 'Predicciones = estimaciones estadísticas. No garantizan resultados futuros.', ''],
            ]
            pd.DataFrame(filas_kpi[1:], columns=filas_kpi[0]).to_excel(
                writer, sheet_name='KPIs y Predicción', index=False)
            if top is not None:
                top.to_excel(writer, sheet_name='Top Productos', index=False)
            if 'mes' in df.columns and 'ventas' in df.columns:
                vm2 = df.groupby('mes').agg(
                    ventas=('ventas','sum'),
                    unidades=('cantidad','sum') if 'cantidad' in df.columns else ('ventas','count'),
                ).reset_index()
                vm2.to_excel(writer, sheet_name='Ventas por Mes', index=False)
        return buf.getvalue()

    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        excel_bytes = generar_excel(df, kpis, top, pred, rango, confianza, modelo_usado)
        st.download_button(
            label="⬇️ Descargar Excel completo",
            data=excel_bytes,
            file_name="salesiq_reporte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with d2:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Exportar datos limpios (.csv)",
            data=csv,
            file_name="datos_limpios.csv",
            mime="text/csv",
        )
    with d3:
        st.markdown("""
        <div style="padding-top:10px;font-size:12px;color:#94a3b8;line-height:1.8;">
            El Excel incluye 4 hojas: datos procesados · KPIs con predicción ·
            top productos · ventas por mes. Listo para entregar al cliente.
        </div>
        """, unsafe_allow_html=True)
