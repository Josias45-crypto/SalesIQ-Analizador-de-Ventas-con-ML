# 📊 SalesIQ — Analizador de Ventas con ML

## ¿Qué hace este programa?

Sube el Excel de ventas de cualquier negocio y en segundos obtienes:

- **KPIs automáticos**: ventas totales, promedio mensual, unidades, márgenes
- **Predicción ML**: el próximo mes de ventas usando Scikit-learn
- **Top productos**: ranking con barra de progreso y ganancias
- **Alertas de stock**: productos con bajo movimiento
- **Insights automáticos**: tendencias comparando mes a mes
- **Reporte descargable**: Excel con todo el análisis

---

## 🚀 Instalación (solo la primera vez)

Abre tu terminal o cmd y ejecuta esto paso a paso:

```bash
# 1. Entra a la carpeta del proyecto
cd sales_analyzer

# 2. (Recomendado) Crea un entorno virtual
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Mac/Linux:
source venv/bin/activate

# 3. Instala las librerías
pip install -r requirements.txt
```

---

## ▶️ Cómo ejecutar la app

```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador en: **http://localhost:8501**

---

## 📁 Estructura del proyecto

```
sales_analyzer/
├── app.py              ← La aplicación principal
├── generar_datos.py    ← Genera un Excel de ejemplo para probar
├── requirements.txt    ← Librerías necesarias
└── README.md           ← Esta guía
```

---

## 📊 Formato del Excel del cliente

El programa acepta archivos con estas columnas (los nombres pueden variar):

| Columna | Nombres aceptados | Obligatorio |
|---------|-------------------|-------------|
| Fecha | Fecha, Date, Mes | ✅ Sí |
| Producto | Producto, Product, Item, Articulo | ✅ Sí |
| Ventas | Ventas, Sales, Total, Monto, Ingresos | ✅ Sí |
| Cantidad | Cantidad, Qty, Units, Unidades | Recomendado |
| Costo | Costo, Cost | Opcional (para margen) |
| Categoría | Categoria, Category, Tipo | Opcional |

---

## 💰 Cómo monetizarlo

### Precio sugerido por servicio:
- **Análisis inicial** (una sola vez): $50 - $100
- **Suscripción mensual** (reportes cada mes): $30 - $80/mes
- **Instalación + capacitación**: $150 - $300

### A quién venderlo:
- 🏪 Tiendas de ropa, calzado, accesorios
- 🍕 Restaurantes y cafeterías
- 💊 Farmacias y distribuidoras
- 🔧 Ferreterías y materiales
- 🛒 Cualquier negocio con Excel de ventas

### Tu pitch de venta:
> *"¿Sabés cuál es tu producto más rentable y cuánto vas a vender el mes que viene? 
> Con mi herramienta lo sabés en 30 segundos. Solo me das tu Excel de ventas."*

---

## 🧠 Librerías que usa (para tu clase)

| Librería | Para qué se usa |
|----------|----------------|
| **Pandas** | Leer, limpiar y transformar el Excel del cliente |
| **NumPy** | Calcular KPIs (suma, promedio, máximo, etc.) |
| **Scikit-learn** | `LinearRegression` para predecir ventas futuras |
| **Plotly** | Gráficas interactivas (barras, líneas, pie) |
| **Streamlit** | La interfaz web completa |
| **OpenPyXL** | Generar el reporte Excel descargable |

---

## 🔧 Generar datos de ejemplo

```bash
python generar_datos.py
```

Esto crea `ventas_ejemplo.xlsx` con 12 meses de datos de una tienda de ropa ficticia.

---

## ❓ Problemas comunes

**Error: "No module named streamlit"**
→ Ejecuta: `pip install -r requirements.txt`

**La app no abre en el navegador**
→ Ve manualmente a: http://localhost:8501

**Error al leer el Excel**
→ Verifica que las columnas tengan nombres similares a los de la tabla de arriba
→ Usa el botón "Usar datos de ejemplo" para probar primero
