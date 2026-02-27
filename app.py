# ============================================================
# SALESIQ PRO — Analizador de Ventas con Machine Learning
# ============================================================
# Estructura:
#   [1] Importaciones y configuración
#   [2] CSS — paleta light profesional
#   [3] Funciones de datos (cargar_datos, calcular_kpis)
#   [4] Función ML (predecir_ventas)
#   [5] Análisis de productos (analisis_productos, detectar_alertas)
#   [6] Datos de ejemplo
#   [7] Sidebar
#   [8] Dashboard principal
# ============================================================

# ── [1] IMPORTACIONES ──────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# ── [2] CSS — PALETA LIGHT PROFESIONAL ─────────────────────
# Dirección: "Consultora de alto nivel"
#   Fondo crema suave, tipografía editorial,
#   azul índigo como acento principal, ámbar para finanzas
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #f7f6f3;
    --surface:   #ffffff;
    --surface2:  #f2f1ee;
    --border:    #e8e6e1;
    --border2:   #d4d1ca;
    --ink:       #1a1916;
    --ink2:      #3d3b36;
    --muted:     #8c8880;
    --muted2:    #b5b2ab;
    --indigo:    #3d52a0;
    --indigo2:   #6272c3;
    --indigo-bg: #eef0f8;
    --amber:     #b8621a;
    --amber-bg:  #fdf3e8;
    --jade:      #1e6b4a;
    --jade-bg:   #e8f5ee;
    --rose:      #b91c3c;
    --rose-bg:   #fde8ed;
    --shadow:    0 1px 3px rgba(26,25,22,0.06), 0 1px 2px rgba(26,25,22,0.04);
    --shadow-md: 0 4px 12px rgba(26,25,22,0.08), 0 2px 4px rgba(26,25,22,0.04);
}

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--ink2) !important; font-family: 'Outfit', sans-serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color: var(--ink) !important; }
[data-testid="stSidebar"] label { font-size: 12px !important; color: var(--muted) !important; }

/* Botón CTA */
.stButton > button {
    background: var(--indigo) !important; color: white !important; border: none !important;
    border-radius: 8px !important; padding: 11px 20px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; width: 100% !important; transition: all 0.18s !important;
    box-shadow: var(--shadow) !important;
}
.stButton > button:hover { background: #2d3f80 !important; transform: translateY(-1px) !important; box-shadow: var(--shadow-md) !important; }

/* Botón descarga */
.stDownloadButton > button {
    background: var(--surface) !important; color: var(--ink) !important;
    border: 1px solid var(--border2) !important; border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 500 !important;
    font-size: 13px !important; width: 100% !important; transition: all 0.18s !important; box-shadow: var(--shadow) !important;
}
.stDownloadButton > button:hover { border-color: var(--indigo) !important; color: var(--indigo) !important; background: var(--indigo-bg) !important; }

/* Hero */
.hero { background: var(--ink); border-radius: 16px; padding: 44px 52px; margin-bottom: 28px; position: relative; overflow: hidden; }
.hero::before { content:''; position:absolute; right:-60px; top:-60px; width:280px; height:280px; background:radial-gradient(circle,rgba(61,82,160,0.4) 0%,transparent 65%); pointer-events:none; }
.hero::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--indigo2),var(--amber),transparent); }
.hero-badge { display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:rgba(255,255,255,0.55) !important; padding:5px 14px; border-radius:20px; font-size:10px; letter-spacing:2px; margin-bottom:18px; font-family:'JetBrains Mono',monospace; text-transform:uppercase; }
.hero-badge::before { content:''; width:5px; height:5px; background:#68d391; border-radius:50%; animation:blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.hero h1 { font-family:'DM Serif Display',serif !important; font-size:clamp(2rem,4vw,3rem) !important; font-weight:400 !important; color:white !important; margin:0 0 10px !important; line-height:1.05 !important; letter-spacing:-0.5px; }
.hero h1 em { font-style:italic; color:rgba(255,255,255,0.65); }
.hero p { color:rgba(255,255,255,0.4) !important; font-size:13px !important; margin:0; font-family:'JetBrains Mono',monospace; }

/* KPI Cards */
.kpi-card { background:var(--surface); border-radius:12px; padding:22px 20px 20px; border:1px solid var(--border); box-shadow:var(--shadow); transition:box-shadow 0.2s,transform 0.2s; height:100%; position:relative; overflow:hidden; }
.kpi-card:hover { box-shadow:var(--shadow-md); transform:translateY(-2px); }
.kpi-top-bar { position:absolute; top:0; left:0; right:0; height:3px; border-radius:12px 12px 0 0; }
.kpi-label { font-size:10px; font-weight:600; letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin-bottom:12px; font-family:'JetBrains Mono',monospace; }
.kpi-value { font-family:'DM Serif Display',serif; font-size:2rem; color:var(--ink); line-height:1; margin-bottom:6px; }
.kpi-sub { font-size:12px; color:var(--muted); }

/* Títulos de sección */
.sec-title { font-family:'DM Serif Display',serif; font-size:1.35rem; font-style:italic; color:var(--ink); margin:36px 0 18px; padding-bottom:12px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; }

/* Predicción */
.pred-card { background:var(--indigo); border-radius:12px; padding:28px 22px; text-align:center; height:100%; position:relative; overflow:hidden; box-shadow:var(--shadow-md); }
.pred-card::before { content:''; position:absolute; right:-40px; bottom:-40px; width:160px; height:160px; background:rgba(255,255,255,0.05); border-radius:50%; }
.pred-tag { font-size:9px; letter-spacing:3px; color:rgba(255,255,255,0.5); text-transform:uppercase; margin-bottom:14px; font-family:'JetBrains Mono',monospace; }
.pred-value { font-family:'DM Serif Display',serif; font-size:2.4rem; color:white; line-height:1; margin-bottom:6px; }
.pred-range { font-size:11px; color:rgba(255,255,255,0.5); margin-bottom:18px; font-family:'JetBrains Mono',monospace; }
.pred-model { font-size:10px; color:rgba(255,255,255,0.35); font-family:'JetBrains Mono',monospace; letter-spacing:1px; margin-top:12px; }

/* Productos */
.prod-row { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:6px; display:grid; grid-template-columns:34px 1fr auto; gap:14px; align-items:center; transition:border-color 0.15s,box-shadow 0.15s; }
.prod-row:hover { border-color:var(--border2); box-shadow:var(--shadow); }
.rank-badge { width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-family:'Outfit',sans-serif; font-weight:700; font-size:12px; }

/* Alertas */
.alert-item { border-radius:8px; padding:11px 14px; margin-bottom:6px; border-left:3px solid; display:flex; justify-content:space-between; align-items:center; font-size:12px; }
.alert-tag { font-size:9px; font-weight:700; letter-spacing:1.5px; padding:3px 8px; border-radius:20px; text-transform:uppercase; white-space:nowrap; margin-left:10px; }

/* Disclaimer */
.disclaimer { background:var(--amber-bg); border:1px solid rgba(184,98,26,0.2); border-left:3px solid var(--amber); border-radius:0 8px 8px 0; padding:12px 16px; font-size:12px; color:var(--amber); line-height:1.7; margin:16px 0; font-family:'JetBrains Mono',monospace; }

/* Bienvenida */
.welcome-card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:60px 48px; text-align:center; box-shadow:var(--shadow); }
.welcome-title { font-family:'DM Serif Display',serif; font-size:2rem; font-style:italic; color:var(--ink); margin-bottom:14px; }
.welcome-sub { color:var(--muted); font-size:13px; line-height:2; max-width:480px; margin:0 auto; }

/* Feature cards */
.feat-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:28px 22px; transition:box-shadow 0.2s,transform 0.2s; box-shadow:var(--shadow); height:100%; }
.feat-card:hover { box-shadow:var(--shadow-md); transform:translateY(-2px); }
.feat-icon { font-size:1.6rem; margin-bottom:14px; display:block; }
.feat-title { font-family:'DM Serif Display',serif; font-size:1.05rem; color:var(--ink); margin-bottom:8px; }
.feat-desc { font-size:12px; color:var(--muted); line-height:1.8; }

/* Sidebar logo/secciones */
.sidebar-logo { padding:28px 20px 20px; border-bottom:1px solid var(--border); margin-bottom:20px; }
.sidebar-logo-name { font-family:'DM Serif Display',serif !important; font-size:1.3rem !important; font-style:italic; color:var(--ink) !important; display:block; margin-bottom:2px; }
.sidebar-logo-tag { font-size:9px !important; letter-spacing:2px !important; color:var(--indigo) !important; font-family:'JetBrains Mono',monospace !important; text-transform:uppercase; }
.sidebar-section { font-size:9px !important; letter-spacing:2.5px !important; color:var(--muted2) !important; text-transform:uppercase; font-family:'JetBrains Mono',monospace !important; margin:18px 0 8px; display:block; }
.sidebar-legal { background:var(--amber-bg); border-left:2px solid rgba(184,98,26,0.4); border-radius:0 6px 6px 0; padding:12px 14px; margin-top:4px; }
.export-info { padding:14px 16px; font-size:12px; color:var(--muted); line-height:1.9; font-family:'JetBrains Mono',monospace; border-left:2px solid var(--border2); }
</style>
""", unsafe_allow_html=True)

# ── [3] FUNCIONES DE DATOS ──────────────────────────────────

def cargar_datos(archivo):
    """
    Lee un Excel/CSV y normaliza las columnas.
    Pasos: leer → normalizar nombres → mapeo flexible
           → parsear fechas → convertir numéricos → calcular ganancia/margen
    """
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo, encoding='utf-8', encoding_errors='replace')
        else:
            df = pd.read_excel(archivo)

        df.columns = df.columns.str.strip().str.lower()

        # Mapeo flexible: acepta múltiples nombres para cada columna estándar
        mapeo = {
            'producto':  ['producto','product','item','articulo','artículo','descripcion','nombre_producto'],
            'fecha':     ['fecha','date','mes','periodo','fecha_venta','date_sale'],
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

        # Parsear fechas y crear columnas de tiempo derivadas
        if 'fecha' in df.columns:
            df['fecha']      = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)
            df               = df.dropna(subset=['fecha'])
            df['mes']        = df['fecha'].dt.to_period('M').astype(str)  # "2024-03"
            df['mes_num']    = df['fecha'].dt.month
            df['anio']       = df['fecha'].dt.year
            df['trimestre']  = df['fecha'].dt.quarter
            df['dia_semana'] = df['fecha'].dt.dayofweek

        # Convertir a numérico (errors='coerce' → texto inválido se vuelve NaN)
        for col in ['ventas', 'cantidad', 'costo', 'precio']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Calcular ganancia y margen si hay columna de costo
        if 'ventas' in df.columns and 'costo' in df.columns:
            df['ganancia']   = df['ventas'] - df['costo']
            df['margen_pct'] = np.where(df['ventas'] > 0, (df['ganancia'] / df['ventas']) * 100, 0)

        if 'ventas' in df.columns:
            df = df[df['ventas'] > 0].reset_index(drop=True)

        return df, None
    except Exception as e:
        return None, str(e)


def calcular_kpis(df):
    """
    KPIs del negocio usando NumPy (operaciones vectorizadas sobre arrays).
    Retorna dict con: total_ventas, promedio_mensual, total_unidades,
    total_ganancia, margen_promedio, n_productos, etc.
    """
    kpis = {}
    if 'ventas' in df.columns:
        v = df['ventas'].values  # Pandas → NumPy array
        kpis['total_ventas']   = np.sum(v)
        kpis['venta_promedio'] = np.mean(v)
        kpis['venta_mediana']  = np.median(v)
        kpis['venta_max']      = np.max(v)
        kpis['variabilidad']   = np.std(v)
        if 'mes' in df.columns:
            vxm = df.groupby('mes')['ventas'].sum().values
            kpis['promedio_mensual'] = np.mean(vxm)
            kpis['mejor_mes_valor']  = np.max(vxm)
            kpis['peor_mes_valor']   = np.min(vxm)
            kpis['n_meses']          = len(vxm)
    if 'cantidad'  in df.columns: kpis['total_unidades']  = int(np.sum(df['cantidad'].values))
    if 'ganancia'  in df.columns:
        kpis['total_ganancia']  = np.sum(df['ganancia'].values)
        kpis['margen_promedio'] = np.mean(df['margen_pct'].values)
    if 'producto'  in df.columns: kpis['n_productos'] = df['producto'].nunique()
    return kpis


# ── [4] MACHINE LEARNING ────────────────────────────────────

def predecir_ventas(df):
    """
    Entrena 3 modelos y elige el más preciso (menor MAPE) para
    predecir las ventas del próximo mes.

    Modelos: Regresión Lineal, Random Forest, Gradient Boosting
    Features: índice de tiempo (tendencia), mes calendario (estacionalidad), trimestre
    Evaluación: cross_val_score con 3 folds para evitar overfitting
    """
    if 'mes' not in df.columns or 'ventas' not in df.columns:
        return None, None, None, "Sin columna fecha"

    vm = df.groupby('mes')['ventas'].sum().reset_index().sort_values('mes')
    if len(vm) < 4:
        return None, None, None, "Se necesitan mínimo 4 meses de datos"

    n          = len(vm)
    tiempo     = np.arange(n)
    meses_cal  = np.array([int(m.split('-')[1]) for m in vm['mes']])
    trimestres = np.ceil(meses_cal / 3).astype(int)

    X        = np.column_stack([tiempo, meses_cal, trimestres])
    y        = vm['ventas'].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelos = {
        'Regresión Lineal':  LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    mejor_modelo, mejor_nombre, mejor_error = None, '', float('inf')

    for nombre, modelo in modelos.items():
        try:
            cv     = cross_val_score(modelo, X_scaled, y, cv=min(3, n-1), scoring='neg_mean_absolute_percentage_error')
            e_cv   = -np.mean(cv)
            modelo.fit(X_scaled, y)
            e_tr   = mean_absolute_percentage_error(y, modelo.predict(X_scaled))
            e_tot  = (e_cv + e_tr) / 2
            if e_tot < mejor_error:
                mejor_error, mejor_modelo, mejor_nombre = e_tot, modelo, nombre
        except:
            continue

    if mejor_modelo is None:
        return None, None, None, "Error al entrenar modelos"

    sig_idx  = n
    sig_mes  = (meses_cal[-1] % 12) + 1
    sig_trim = int(np.ceil(sig_mes / 3))
    pred     = max(0, mejor_modelo.predict(scaler.transform([[sig_idx, sig_mes, sig_trim]]))[0])

    # Intervalo de confianza basado en el error histórico del modelo
    errores    = np.abs(y - mejor_modelo.predict(X_scaled))
    margen     = np.mean(errores) * 1.5
    confianza  = max(0, min(95, (1 - mejor_error) * 100))

    return pred, (max(0, pred - margen), pred + margen), confianza, mejor_nombre


# ── [5] ANÁLISIS DE PRODUCTOS ───────────────────────────────

def analisis_productos(df, n=10):
    """
    Ranking top-N productos por ventas.
    Calcula: ventas totales, ticket promedio, unidades, ganancia, margen.
    """
    if 'producto' not in df.columns or 'ventas' not in df.columns:
        return None

    agg = {'ventas': ['sum', 'mean', 'count']}
    if 'cantidad'   in df.columns: agg['cantidad']   = 'sum'
    if 'ganancia'   in df.columns: agg['ganancia']   = 'sum'
    if 'margen_pct' in df.columns: agg['margen_pct'] = 'mean'

    top = df.groupby('producto').agg(agg)
    top.columns = ['_'.join(c).strip('_') for c in top.columns]
    top = top.rename(columns={
        'ventas_sum': 'ventas', 'ventas_mean': 'ticket_prom', 'ventas_count': 'transacciones',
        'cantidad_sum': 'unidades', 'ganancia_sum': 'ganancia', 'margen_pct_mean': 'margen'
    })
    return top.sort_values('ventas', ascending=False).head(n).reset_index()


def detectar_alertas(df, umbral_stock=10, umbral_caida=0.20):
    """
    Detecta automáticamente:
    - Crítico: bajo stock este mes, caída > umbral_caida vs mes anterior
    - Advertencia: stock bajo moderado
    - Positivo: crecimiento > 20% vs mes anterior
    """
    alertas = {'critico': [], 'advertencia': [], 'positivo': []}
    if 'producto' not in df.columns:
        return alertas

    # Alerta de stock bajo en el último mes
    if 'mes' in df.columns and 'cantidad' in df.columns:
        ult = df['mes'].max()
        rec = df[df['mes'] == ult].groupby('producto')['cantidad'].sum()
        for prod, cant in rec[rec < umbral_stock].items():
            nivel = 'critico' if cant < umbral_stock // 2 else 'advertencia'
            alertas[nivel].append(f"{prod}: solo {cant:.0f} uds. este mes")

    # Alerta de variación mes a mes
    if 'mes' in df.columns and 'ventas' in df.columns:
        vm = df.groupby(['producto', 'mes'])['ventas'].sum().unstack(fill_value=0)
        if vm.shape[1] >= 2:
            cambio = np.where(vm.iloc[:, -2] > 0, (vm.iloc[:, -1] - vm.iloc[:, -2]) / vm.iloc[:, -2], 0)
            for i, prod in enumerate(vm.index):
                if cambio[i] < -umbral_caida:
                    alertas['critico'].append(f"{prod}: ↓{abs(cambio[i])*100:.0f}% vs mes anterior")
                elif cambio[i] > 0.20:
                    alertas['positivo'].append(f"{prod}: ↑{cambio[i]*100:.0f}% vs mes anterior")

    return alertas


# ── [6] DATOS DE EJEMPLO ────────────────────────────────────

@st.cache_data
def datos_ejemplo():
    """Dataset sintético de tienda de ropa: 10 productos, 12 meses, con tendencia y estacionalidad."""
    np.random.seed(42)
    productos = [
        ('Camisa Slim Fit','Tops',18,45), ('Jeans Clásico','Pantalones',25,65),
        ('Vestido Floral','Vestidos',22,58), ('Chaqueta Cuero','Outerwear',55,120),
        ('Zapatos Oxford','Calzado',35,89), ('Bolso Cuero','Accesorios',45,110),
        ('Gorra Casual','Accesorios',8,22), ('Sudadera Hoodie','Tops',20,55),
        ('Pantalón Chino','Pantalones',22,55), ('Blusa Elegante','Tops',16,42),
    ]
    filas = []
    for m in range(1, 13):
        fb = pd.Timestamp(f'2024-{m:02d}-01')
        for prod, cat, costo, precio in productos:
            tend  = 1 + m * 0.025
            estac = 1.4 if m in [6,7,12] else 0.80 if m in [1,2] else 1.0
            cant  = max(1, int(np.random.poisson(18 * tend * estac)))
            filas.append({'Fecha': fb + pd.Timedelta(days=int(np.random.randint(0,27))),
                          'Producto': prod, 'Categoria': cat,
                          'Cantidad': cant, 'Ventas': cant*precio, 'Costo': cant*costo})
    return pd.DataFrame(filas)

# ── [7] SIDEBAR ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-name">SalesIQ Pro</span>
        <span class="sidebar-logo-tag">▸ ML Analytics Engine</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-section">Fuente de datos</span>', unsafe_allow_html=True)
    archivo      = st.file_uploader("Excel o CSV de ventas", type=['xlsx','xls','csv'],
                                    help="Columnas: Fecha, Producto, Ventas, Cantidad, Costo, Categoria")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    usar_ejemplo = st.button("▶  Cargar datos de ejemplo")

    st.markdown("---")
    st.markdown('<span class="sidebar-section">Configuración</span>', unsafe_allow_html=True)
    top_n        = st.slider("Top productos", 3, 20, 10)
    umbral_stock = st.slider("Alerta stock (uds.)", 1, 50, 10)
    umbral_caida = st.slider("Alerta caída de ventas (%)", 5, 50, 20) / 100

    st.markdown("---")
    st.markdown("""
    <span class="sidebar-section">Columnas aceptadas</span>
    <div style='font-size:11px;color:#b5b2ab;line-height:2.2;font-family:JetBrains Mono,monospace;'>
        → Fecha / Date / Mes<br>
        → Producto / Item<br>
        → Ventas / Sales / Total<br>
        → Cantidad / Qty<br>
        → Costo / Cost <span style='opacity:0.5;'>(opcional)</span><br>
        → Categoria / Category <span style='opacity:0.5;'>(opcional)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-legal">
        <div style='font-size:9px;letter-spacing:2px;color:#b8621a;margin-bottom:6px;font-family:JetBrains Mono,monospace;'>⚠ AVISO LEGAL</div>
        <div style='font-size:11px;color:#b8621a;line-height:1.8;font-family:JetBrains Mono,monospace;opacity:0.8;'>
            Predicciones estadísticas basadas en histórico. No garantizan resultados futuros.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── [8] INTERFAZ PRINCIPAL ──────────────────────────────────

# Hero siempre visible
st.markdown("""
<div class="hero">
    <div class="hero-badge">Powered by Scikit-Learn + Pandas</div>
    <h1>Sales<em>IQ</em> Pro</h1>
    <p>// Inteligencia de ventas con Machine Learning — decisiones basadas en datos reales</p>
</div>
""", unsafe_allow_html=True)

# Carga de datos
df = None
if archivo:
    df, error = cargar_datos(archivo)
    if error:
        st.error(f"❌ Error al leer el archivo: {error}")
        df = None
elif usar_ejemplo:
    import io
    df_raw = datos_ejemplo()
    buf    = io.BytesIO()
    df_raw.to_excel(buf, index=False)
    buf.seek(0); buf.name = 'ejemplo.xlsx'
    df, _ = cargar_datos(buf)
    if df is None:
        df = df_raw.copy()
        df.columns     = df.columns.str.lower()
        df['fecha']    = pd.to_datetime(df['fecha'])
        df['mes']      = df['fecha'].dt.to_period('M').astype(str)
        df['ventas']   = df['ventas'].astype(float)
        df['cantidad'] = df['cantidad'].astype(float)
        df['costo']    = df['costo'].astype(float)
        df['ganancia'] = df['ventas'] - df['costo']
        df['margen_pct'] = (df['ganancia'] / df['ventas']) * 100

# ── PANTALLA DE BIENVENIDA (sin datos)
if df is None:
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:3rem;margin-bottom:20px;">📂</div>
        <div class="welcome-title">Sube tu archivo para comenzar</div>
        <div class="welcome-sub">
            Acepta <strong>Excel (.xlsx)</strong> o <strong>CSV</strong>.<br>
            ¿No tienes un archivo? Usa <strong style="color:#3d52a0;">▶ Cargar datos de ejemplo</strong> desde el panel lateral.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🔮", "Predicción ML",       "3 modelos compiten; el más preciso predice el próximo mes con intervalo de confianza."),
        (c2, "📊", "Análisis profundo",    "KPIs, top productos, márgenes, tendencias mensuales y distribución por categoría."),
        (c3, "⚠️", "Alertas inteligentes", "Detecta caídas de ventas, bajo stock y productos estrella automáticamente."),
    ]:
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <span class="feat-icon">{icon}</span>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── DASHBOARD PRINCIPAL (con datos)
else:
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Aviso:</strong> Las predicciones son estimaciones estadísticas basadas en datos históricos.
        No representan ni garantizan resultados futuros. Úselas como referencia de apoyo.
    </div>
    """, unsafe_allow_html=True)

    # Calcular todo antes de renderizar
    kpis                                 = calcular_kpis(df)
    pred, rango, confianza, modelo_usado = predecir_ventas(df)
    top                                  = analisis_productos(df, n=top_n)
    alertas                              = detectar_alertas(df, umbral_stock, umbral_caida)
    vm = df.groupby('mes')['ventas'].sum().reset_index().sort_values('mes') if 'mes' in df.columns else None

    # ── BLOQUE A: KPIs
    st.markdown('<div class="sec-title">📌 Resumen ejecutivo</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_items = [
        (k1, "VENTAS TOTALES",   f"${kpis.get('total_ventas',0):,.0f}",   f"{kpis.get('n_meses','?')} meses analizados", "#3d52a0"),
        (k2, "PROMEDIO MENSUAL", f"${kpis.get('promedio_mensual',0):,.0f}", "por mes de operación", "#6272c3"),
        (k3, "UNIDADES VENDIDAS",f"{kpis.get('total_unidades',0):,}",      "total de piezas", "#1e6b4a"),
        (k4, "GANANCIA NETA",
         f"${kpis.get('total_ganancia',0):,.0f}" if 'total_ganancia' in kpis else f"{kpis.get('n_productos','?')} prods.",
         f"margen {kpis.get('margen_promedio',0):.1f}%" if 'margen_promedio' in kpis else "productos únicos",
         "#b8621a"),
        (k5, "PREDICCIÓN ML",
         f"${pred:,.0f}" if pred else "—",
         f"próximo mes · {confianza:.0f}% conf." if pred else "necesitas +4 meses",
         "#b91c3c"),
    ]
    for col, label, valor, sub, color in kpi_items:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:{color};"></div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{valor}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── BLOQUE B: Tendencia + Predicción
    st.markdown('<div class="sec-title">📈 Tendencia de ventas</div>', unsafe_allow_html=True)
    col_graf, col_pred = st.columns([3, 1])

    with col_graf:
        if vm is not None and len(vm) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=vm['mes'], y=vm['ventas'], name='Ventas reales',
                                 marker_color='#eef0f8', marker_line_color='#c8cfe8', marker_line_width=1))
            fig.add_trace(go.Scatter(x=vm['mes'], y=vm['ventas'], mode='lines+markers', name='Tendencia',
                                     line=dict(color='#3d52a0', width=2.5),
                                     marker=dict(size=7, color='white', line=dict(color='#3d52a0', width=2))))
            if pred and rango:
                sig = (pd.Period(vm['mes'].iloc[-1], 'M') + 1).strftime('%Y-%m')
                fig.add_trace(go.Scatter(x=[sig], y=[pred], mode='markers', name='Predicción',
                                         marker=dict(size=16, color='#b8621a', symbol='diamond',
                                                     line=dict(color='white', width=2)),
                                         error_y=dict(type='data', array=[rango[1]-pred],
                                                      arrayminus=[pred-rango[0]], visible=True,
                                                      color='#b8621a', thickness=2)))
            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='white', font_family='Outfit', font_color='#8c8880',
                margin=dict(l=10,r=10,t=10,b=10), height=320,
                legend=dict(orientation='h', y=-0.2, font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(showgrid=False, tickfont=dict(size=11), linecolor='#e8e6e1'),
                yaxis=dict(showgrid=True, gridcolor='#f2f1ee', tickprefix='$', tickfont=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_pred:
        if pred and rango:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-tag">◈ Próximo mes</div>
                <div class="pred-value">${pred:,.0f}</div>
                <div class="pred-range">${rango[0]:,.0f} — ${rango[1]:,.0f}</div>
                <div style="font-size:9px;letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:6px;font-family:'JetBrains Mono';">CONFIANZA</div>
                <div style="background:rgba(255,255,255,0.12);border-radius:3px;height:5px;">
                    <div style="background:rgba(255,255,255,0.7);height:5px;border-radius:3px;width:{confianza:.0f}%;"></div>
                </div>
                <div style="font-size:1.1rem;font-family:'DM Serif Display',serif;color:white;margin-top:6px;">{confianza:.0f}%</div>
                <div class="pred-model">▸ {modelo_usado}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pred-card">
                <div class="pred-tag">◈ Predicción</div>
                <div style="color:rgba(255,255,255,0.4);font-size:12px;margin-top:16px;line-height:1.9;font-family:'JetBrains Mono';">
                    Necesitas mínimo 4 meses con columna Fecha para activar el modelo.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── BLOQUE C: Top Productos + Categorías
    st.markdown('<div class="sec-title">🏆 Análisis de productos</div>', unsafe_allow_html=True)
    col_top, col_cat = st.columns([3, 2])

    with col_top:
        if top is not None:
            max_v = top['ventas'].max()
            rank_styles = {0: ('#b8621a','#fdf3e8'), 1: ('#6272c3','#eef0f8'), 2: ('#8c8880','#f7f6f3')}
            for i, row in top.iterrows():
                ctxt, cbg = rank_styles.get(i, ('#8c8880','#f7f6f3'))
                pct       = (row['ventas'] / max_v) * 100
                m_str     = f"· {row['margen']:.0f}% margen" if 'margen' in row and pd.notna(row.get('margen')) else ""
                g_str     = f"<span style='color:#1e6b4a;font-size:11px;font-weight:600;'>+${row['ganancia']:,.0f}</span>" if 'ganancia' in row else ""
                st.markdown(f"""
                <div class="prod-row">
                    <div class="rank-badge" style="background:{cbg};color:{ctxt};">#{i+1}</div>
                    <div>
                        <div style="font-weight:600;color:#1a1916;font-size:13px;">{row['producto']}</div>
                        <div style="background:#f2f1ee;border-radius:2px;height:2px;margin:6px 0;">
                            <div style="background:#3d52a0;height:2px;border-radius:2px;width:{pct:.0f}%;opacity:0.5;"></div>
                        </div>
                        <div style="font-size:10px;color:#b5b2ab;font-family:'JetBrains Mono',monospace;">{m_str}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1a1916;">${row['ventas']:,.0f}</div>
                        <div>{g_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_cat:
        if 'categoria' in df.columns:
            vc  = df.groupby('categoria')['ventas'].sum().sort_values(ascending=False)
            fig2 = go.Figure(go.Pie(
                labels=vc.index, values=vc.values, hole=0.60,
                marker=dict(colors=['#3d52a0','#b8621a','#1e6b4a','#6272c3','#b91c3c','#8c8880']),
                textfont=dict(family='Outfit', size=12, color='#3d3b36'),
                textinfo='label+percent',
            ))
            fig2.update_layout(paper_bgcolor='white', font_family='Outfit',
                               margin=dict(l=10,r=10,t=10,b=10), showlegend=False, height=360)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#f7f6f3;border:1px solid #e8e6e1;border-radius:10px;padding:32px;
                 text-align:center;color:#b5b2ab;font-size:12px;font-family:'JetBrains Mono',monospace;">
                Agrega columna <strong>Categoria</strong> para ver distribución.
            </div>""", unsafe_allow_html=True)

    # ── BLOQUE D: Alertas
    st.markdown('<div class="sec-title">⚠️ Alertas del negocio</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    alert_cfg = [
        (a1, "critico",     "● Crítico",     "#b91c3c", "#fde8ed", "URGENTE"),
        (a2, "advertencia", "◐ Advertencia", "#b8621a", "#fdf3e8", "REVISAR"),
        (a3, "positivo",    "● Positivo",    "#1e6b4a", "#e8f5ee", "↑ OK"),
    ]
    sin_msg = {"critico": "✓ Sin alertas críticas", "advertencia": "✓ Sin advertencias", "positivo": "Sin variaciones positivas"}
    for col, key, title, color, bg, tag in alert_cfg:
        with col:
            st.markdown(f"<div style='font-size:11px;font-weight:600;color:{color};letter-spacing:1px;margin-bottom:10px;'>{title}</div>", unsafe_allow_html=True)
            if alertas[key]:
                for a in alertas[key][:5]:
                    st.markdown(f"""
                    <div class="alert-item" style="background:{bg};border-left-color:{color};">
                        <span style="color:#3d3b36;">{a}</span>
                        <span class="alert-tag" style="color:{color};background:rgba(0,0,0,0.06);">{tag}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-item" style="background:#f7f6f3;border-left-color:#e8e6e1;color:#b5b2ab;">
                    {sin_msg[key]}
                </div>""", unsafe_allow_html=True)

    # ── BLOQUE E: Exportar
    st.markdown('<div class="sec-title">📁 Exportar reporte</div>', unsafe_allow_html=True)

    def generar_excel(df, kpis, top, pred, rango, confianza, modelo):
        """Genera Excel con 4 hojas: Datos Procesados, KPIs+Predicción, Top Productos, Ventas por Mes."""
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, sheet_name='Datos Procesados', index=False)
            filas = [
                ['Métrica','Valor'],
                ['Total Ventas',       f"${kpis.get('total_ventas',0):,.2f}"],
                ['Promedio Mensual',   f"${kpis.get('promedio_mensual',0):,.2f}"],
                ['Total Unidades',     f"{kpis.get('total_unidades',0):,}"],
                ['Ganancia Total',     f"${kpis.get('total_ganancia',0):,.2f}" if 'total_ganancia' in kpis else 'N/A'],
                ['Margen Promedio',    f"{kpis.get('margen_promedio',0):.1f}%" if 'margen_promedio' in kpis else 'N/A'],
                ['',''],
                ['── PREDICCIÓN ML ──',''],
                ['Estimación próximo mes', f"${pred:,.2f}" if pred else 'N/A'],
                ['Rango mínimo',       f"${rango[0]:,.2f}" if rango else 'N/A'],
                ['Rango máximo',       f"${rango[1]:,.2f}" if rango else 'N/A'],
                ['Confianza',          f"{confianza:.0f}%" if confianza else 'N/A'],
                ['Modelo',             modelo or 'N/A'],
                ['',''],
                ['⚠ AVISO LEGAL','Predicciones estadísticas. No garantizan resultados futuros.'],
            ]
            pd.DataFrame(filas[1:], columns=filas[0]).to_excel(w, sheet_name='KPIs y Predicción', index=False)
            if top is not None:
                top.to_excel(w, sheet_name='Top Productos', index=False)
            if 'mes' in df.columns:
                df.groupby('mes').agg(ventas=('ventas','sum')).reset_index().to_excel(w, sheet_name='Ventas por Mes', index=False)
        return buf.getvalue()

    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        st.download_button("⬇  Descargar Excel completo",
                           generar_excel(df, kpis, top, pred, rango, confianza, modelo_usado),
                           "salesiq_reporte.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with d2:
        st.download_button("⬇  Exportar datos limpios (.csv)",
                           df.to_csv(index=False).encode('utf-8'),
                           "datos_limpios.csv", "text/csv")
    with d3:
        st.markdown("""
        <div class="export-info">
            Excel incluye 4 hojas: datos procesados · KPIs con predicción<br>
            top productos · ventas por mes — listo para entregar al cliente.
        </div>""", unsafe_allow_html=True)
