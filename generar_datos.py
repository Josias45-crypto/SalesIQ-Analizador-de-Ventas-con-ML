import pandas as pd
import numpy as np

np.random.seed(42)

productos = ['Camisa Slim Fit', 'Jeans Clásico', 'Vestido Floral', 'Chaqueta Cuero',
             'Zapatos Oxford', 'Bolso Cuero', 'Gorra Casual', 'Sudadera Hoodie',
             'Pantalón Chino', 'Blusa Elegante']
categorias = ['Tops', 'Pantalones', 'Vestidos', 'Outerwear',
              'Calzado', 'Accesorios', 'Accesorios', 'Tops',
              'Pantalones', 'Tops']
costos =  [18, 25, 22, 55, 35, 45, 8, 20, 22, 16]
precios = [45, 65, 58, 120, 89, 110, 22, 55, 55, 42]

filas = []
for mes_num in range(1, 13):
    fecha_base = pd.Timestamp(f'2024-{mes_num:02d}-01')
    for i, prod in enumerate(productos):
        tendencia = 1 + mes_num * 0.03
        estacional = 1.3 if mes_num in [6, 7, 12] else 0.85 if mes_num in [2, 3] else 1.0
        cantidad = int(np.random.poisson(20 * tendencia * estacional))
        venta = cantidad * precios[i]
        costo_total = cantidad * costos[i]
        filas.append({
            'Fecha': fecha_base + pd.Timedelta(days=int(np.random.randint(0, 28))),
            'Producto': prod,
            'Categoria': categorias[i],
            'Cantidad': cantidad,
            'Ventas': venta,
            'Costo': costo_total
        })

df = pd.DataFrame(filas)
df.to_excel('ventas_ejemplo.xlsx', index=False)
print(f"✅ Archivo creado: ventas_ejemplo.xlsx ({len(df)} registros)")
print(df.head(10))
