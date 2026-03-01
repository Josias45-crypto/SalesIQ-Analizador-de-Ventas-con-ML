import pandas as pd

PLANES = {
    "basico":      3,
    "pro":         12,
    "empresarial": 9999,
}

def filtrar_por_plan(df, plan):
    """
    Recorta el DataFrame según los meses permitidos por el plan.
    """
    meses_permitidos = PLANES.get(plan, 3)

    if "mes" not in df.columns:
        return df

    meses_disponibles = sorted(df["mes"].unique())
    if len(meses_disponibles) <= meses_permitidos:
        return df

    # Tomar solo los últimos N meses permitidos
    meses_a_usar = meses_disponibles[-meses_permitidos:]
    df_filtrado = df[df["mes"].isin(meses_a_usar)].copy()
    return df_filtrado

def mensaje_limite(plan, meses_totales):
    """Muestra aviso si el usuario tiene más datos de los que puede usar."""
    meses_permitidos = PLANES.get(plan, 3)
    if meses_totales > meses_permitidos:
        nombres = {"basico": "Básico", "pro": "Pro", "empresarial": "Empresarial"}
        return f"Tu plan **{nombres[plan]}** incluye hasta **{meses_permitidos} meses** de historial. Tienes {meses_totales} meses — actualiza tu plan para usar todos tus datos."
    return None
