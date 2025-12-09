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

## Publicar en GitHub
Si quieres subir el proyecto a tu propio repositorio de GitHub, estos son los pasos básicos:

```bash
# 1) Crea un repositorio vacío en tu cuenta de GitHub

# 2) Configura el remoto en este proyecto local
git remote add origin git@github.com:TU_USUARIO/TU_REPO.git

# 3) Sube la rama actual (por ejemplo, main o work)
git push -u origin work

# 4) Opcional: crea un pull request desde GitHub
```

Sustituye `TU_USUARIO/TU_REPO` por tu ruta real. Si prefieres usar HTTPS, reemplaza la URL del remoto por `https://github.com/TU_USUARIO/TU_REPO.git`.
