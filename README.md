# Finance CRM Dashboard (Streamlit + SQLite)

Personal finance dashboard with persistent storage, CRM-style navigation and alerts.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Inicio de sesión demo: **demo@user.com / demo1234**.

## Características
- Autenticación simple por email + contraseña (datos por usuario).
- Configuración por usuario: año, moneda, meta de ahorro.
- Categorías de ingresos y gastos con presupuesto mensual opcional.
- CRUD de transacciones con filtros y exportación CSV.
- Dashboard con KPIs, alertas de presupuesto/ahorro, gráficos y resúmenes.
- Forecast simple de 6 meses usando promedios móviles.
- Datos seed listos (demo user, categorías y movimientos 2026) en SQLite `finance.db`.
