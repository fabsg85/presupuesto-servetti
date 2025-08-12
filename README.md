
# Presupuesto Familiar — Dashboard (Streamlit)

Un dashboard interactivo para el Excel **PRESUPUESTO FAMILIA SERVETTI**. Permite:
- Ver KPIs YTD (ingresos, gastos, ahorro neto, tasa de ahorro).
- Desglose por categorías de gasto (si existen en el Excel).
- Gráficos de evolución (ingresos, gastos, neto) y balance acumulado.
- Forecast de 6 meses del cash flow neto (Holt-Winters).
- CRUD de transacciones manuales (agregar/editar/eliminar) con exportación/importación CSV.

## Estructura de datos
El app parsea los totales e intenta leer el desglose de **GASTOS** por categoría por mes. Las transacciones manuales funcionan como **ajustes** arriba del Excel.

## Deploy gratis (dos opciones)

### Opción A — Streamlit Community Cloud (gratis)
1. Subí estos archivos a un repo público en GitHub: `app.py`, `requirements.txt`.
2. Ingresá a https://share.streamlit.io , conectá tu GitHub y elegí el repo.
3. Configuración:
   - Archivo principal: `app.py`
   - Python: 3.11+
4. Deploy. Abrí la URL que te da Streamlit.

### Opción B — Hugging Face Spaces (gratis)
1. Creá un Space nuevo tipo **Streamlit**.
2. Subí `app.py` y `requirements.txt`.
3. Deploy automático. Listo.

> Persistencia: si querés persistir transacciones sin manejar CSV, podés conectar una Google Sheet usando `gspread` (requiere credenciales).

## Uso
1. Subí el Excel original desde la barra lateral.
2. (Opcional) Cargá/edita transacciones manuales.
3. Explorá KPIs, gráficos y forecast.
4. Descargá CSVs para respaldo.

