
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Presupuesto Familiar — Persistente (Plus)", layout="wide")
st.title("📊 Presupuesto Familiar — Persistente (Plus)")
st.caption("Google Sheets backend + Budget vs Actual + Exportación a Excel + Metas de ahorro.")

# ----------------------------------
# Google Sheets helpers
# ----------------------------------
def get_gs_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        info = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc
    except Exception as e:
        st.error("No se pudo inicializar Google Sheets. Revisá los *Secrets* (gcp_service_account) y compartir el Sheet con el Service Account (Editor).")
        st.stop()

def open_or_create_sheet(gc, sheet_url_or_name):
    try:
        if sheet_url_or_name.startswith("http"):
            sh = gc.open_by_url(sheet_url_or_name)
        else:
            sh = gc.open(sheet_url_or_name)
    except Exception:
        sh = gc.create(sheet_url_or_name)
    return sh

def read_ws_as_df(sh, title, columns):
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=2000, cols=max(10, len(columns)))
        ws.append_row(columns)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    for c in columns:
        if c not in df.columns:
            df[c] = "" if c not in ["amount","month","year","target_amount","monthly_contrib"] else 0
    return df[columns], ws

def write_df_to_ws(ws, df):
    ws.clear()
    ws.update([df.columns.tolist()] + df.fillna("").values.tolist())

# ----------------------------------
# Data model
# ----------------------------------
with st.sidebar:
    st.header("🔐 Conexión")
    sheet_ref = st.text_input("URL o nombre de Google Sheet", value=st.session_state.get("sheet_ref","Presupuesto Familiar Servetti"))
    st.session_state["sheet_ref"] = sheet_ref
    st.caption("Si no existe, se creará automáticamente.")

gc = get_gs_client()
sh = open_or_create_sheet(gc, sheet_ref)

settings_cols = ["key","value"]
income_cols   = ["year","month","category","amount","account","currency","tags"]
expense_cols  = ["year","month","category","amount","account","currency","tags"]
tx_cols       = ["date","type","category","amount","account","currency","tags","notes"]
budget_cols   = ["year","month","category","type","amount"]  # type: Ingreso/Gasto
goals_cols    = ["name","target_amount","target_date","monthly_contrib","priority","notes"]

settings_df, ws_settings = read_ws_as_df(sh, "SETTINGS", settings_cols)
income_df,   ws_income   = read_ws_as_df(sh, "INCOME_CATS", income_cols)
expense_df,  ws_expense  = read_ws_as_df(sh, "EXPENSE_CATS", expense_cols)
tx_df,       ws_tx       = read_ws_as_df(sh, "TRANSACTIONS", tx_cols)
budget_df,   ws_budget   = read_ws_as_df(sh, "BUDGET", budget_cols)
goals_df,    ws_goals    = read_ws_as_df(sh, "GOALS", goals_cols)

# Defaults
if settings_df.empty:
    settings_df = pd.DataFrame({"key":["base_currency","fx_usd_uyu","variance_green","variance_yellow"],
                                "value":["UYU","40","0.05","0.15"]})
    write_df_to_ws(ws_settings, settings_df)

# Parse settings
smap = settings_df.set_index("key")["value"].to_dict()
base_currency = smap.get("base_currency","UYU")
try:
    fx = float(smap.get("fx_usd_uyu","40"))
except:
    fx = 40.0
try:
    variance_green = float(smap.get("variance_green","0.05"))  # 5%
    variance_yellow = float(smap.get("variance_yellow","0.15")) # 15%
except:
    variance_green, variance_yellow = 0.05, 0.15

with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Configuración")
    base_currency = st.selectbox("Moneda base", ["UYU","USD"], index=0 if base_currency=="UYU" else 1)
    fx = st.number_input("Tipo de cambio (USD→UYU)", min_value=1.0, value=float(fx), step=0.5)
    variance_green = st.number_input("Umbral verde (desvío ≤)", min_value=0.0, value=float(variance_green), step=0.01, help="Proporción vs presupuesto. Ej. 0.05 = 5%")
    variance_yellow = st.number_input("Umbral amarillo (desvío ≤)", min_value=0.0, value=float(variance_yellow), step=0.01, help="Rojo si excede este valor.")
    if st.button("💾 Guardar configuración"):
        cfg = pd.DataFrame({"key":["base_currency","fx_usd_uyu","variance_green","variance_yellow"],
                            "value":[base_currency, str(fx), str(variance_green), str(variance_yellow)]})
        write_df_to_ws(ws_settings, cfg)
        st.success("Configuración guardada.")

# ----------------------------------
# Helpers
# ----------------------------------
MONTHS = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
REV_MONTHS = {i+1:m for i,m in enumerate(MONTHS)}
MONTHS_MAP = {m:i+1 for i,m in enumerate(MONTHS)}

def convert_amount(amount, cur_from, base_currency, fx):
    if cur_from == base_currency:
        return amount
    if cur_from == "USD" and base_currency == "UYU":
        return amount * fx
    if cur_from == "UYU" and base_currency == "USD":
        return amount / fx
    return amount

def build_totals(income_df, expense_df, tx_df):
    inc = income_df.copy()
    exp = expense_df.copy()
    if not inc.empty:
        inc["amount_base"] = [convert_amount(a, c, base_currency, fx) for a,c in zip(inc["amount"], inc["currency"])]
    if not exp.empty:
        exp["amount_base"] = [convert_amount(a, c, base_currency, fx) for a,c in zip(exp["amount"], exp["currency"])]
    inc_tot = inc.groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"income"})
    exp_tot = exp.groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"expense"})
    merged = pd.merge(inc_tot, exp_tot, on=["year","month"], how="outer").fillna(0)

    if not tx_df.empty:
        t = tx_df.dropna(subset=["date","amount"]).copy()
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t = t.dropna(subset=["date"])
        t["year"] = t["date"].dt.year.astype(str)
        t["month"] = t["date"].dt.month
        t["amount_base"] = [convert_amount(a, c, base_currency, fx) for a,c in zip(t["amount"], t["currency"])]
        inc_tx = t[t["type"]=="Ingreso"].groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"inc_tx"})
        exp_tx = t[t["type"]=="Gasto"].groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"exp_tx"})
        merged = pd.merge(merged, inc_tx, on=["year","month"], how="left").fillna({"inc_tx":0})
        merged = pd.merge(merged, exp_tx, on=["year","month"], how="left").fillna({"exp_tx":0})
        merged["income"]  = merged["income"] + merged["inc_tx"]
        merged["expense"] = merged["expense"] + merged["exp_tx"]

    merged["net"] = merged["income"] - merged["expense"]
    merged["year"] = merged["year"].astype(str)
    merged["month"] = pd.to_numeric(merged["month"], errors="coerce")
    return merged

# ----------------------------------
# Editores persistentes
# ----------------------------------
st.subheader("🧾 Editores persistentes (ingresos/gastos por categoría)")
years_present = sorted(list(set(income_df["year"].tolist() + expense_df["year"].tolist())))
if not years_present:
    years_present = [datetime.now().year]
year_sel = st.selectbox("Año", years_present, index=len(years_present)-1)

def to_wide(df, year):
    d = df[df["year"].astype(str)==str(year)].copy()
    if d.empty:
        return pd.DataFrame(columns=["name"]+MONTHS)
    d["month"] = pd.to_numeric(d["month"], errors="coerce")
    w = d.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    w.columns = [REV_MONTHS.get(c,c) for c in w.columns]
    w = w.reset_index().rename(columns={"category":"name"})
    for m in MONTHS:
        if m not in w.columns:
            w[m] = 0.0
    return w[["name"]+MONTHS]

inc_wide = to_wide(income_df, year_sel)
exp_wide = to_wide(expense_df, year_sel)

c1,c2 = st.columns(2)
with c1:
    st.markdown("**Ingresos (por categoría y mes)**")
    inc_edit = st.data_editor(inc_wide, num_rows="dynamic", use_container_width=True, key=f"inc_edit_{year_sel}")
with c2:
    st.markdown("**Gastos (por categoría y mes)**")
    exp_edit = st.data_editor(exp_wide, num_rows="dynamic", use_container_width=True, key=f"exp_edit_{year_sel}")

def wide_to_tidy(wide, year):
    rows = []
    for _, r in wide.iterrows():
        for m in MONTHS:
            v = r.get(m, 0.0)
            if pd.notna(v) and float(v) != 0.0:
                rows.append({
                    "year": str(year), "month": MONTHS_MAP[m], "category": r["name"], "amount": float(v),
                    "account":"GS", "currency":"UYU", "tags":""
                })
    return pd.DataFrame(rows, columns=income_cols)

if st.button("💾 Guardar ingresos/gastos del año"):
    inc_new = wide_to_tidy(inc_edit, year_sel)
    exp_new = wide_to_tidy(exp_edit, year_sel).rename(columns={"category":"category","amount":"amount"})
    income_df2 = income_df[income_df["year"].astype(str)!=str(year_sel)].copy()
    expense_df2 = expense_df[expense_df["year"].astype(str)!=str(year_sel)].copy()
    income_df2 = pd.concat([income_df2, inc_new], ignore_index=True)
    expense_df2 = pd.concat([expense_df2, exp_new], ignore_index=True)
    write_df_to_ws(ws_income, income_df2[income_cols])
    write_df_to_ws(ws_expense, expense_df2[expense_cols])
    st.success("Ingresos y gastos guardados.")

# ----------------------------------
# Transacciones y Metas (persistentes)
# ----------------------------------
st.subheader("🧾 Transacciones manuales (persistentes)")
tx_edit = st.data_editor(tx_df, num_rows="dynamic", use_container_width=True, key="tx_persist")
if st.button("💾 Guardar transacciones"):
    df = tx_edit.copy()
    if not df.empty:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        except Exception:
            pass
    write_df_to_ws(ws_tx, df[tx_cols])
    st.success("Transacciones guardadas.")

st.subheader("🎯 Metas de ahorro (GOALS)")
goals_edit = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True, key="goals_persist",
                            column_config={
                                "target_amount": st.column_config.NumberColumn("Monto objetivo", step=100.0, format="%.2f"),
                                "monthly_contrib": st.column_config.NumberColumn("Aporte mensual", step=100.0, format="%.2f"),
                                "target_date": st.column_config.DateColumn("Fecha objetivo"),
                                "priority": st.column_config.SelectboxColumn("Prioridad", options=["alta","media","baja"]),
                                "name": st.column_config.TextColumn("Meta"),
                                "notes": st.column_config.TextColumn("Notas"),
                            })
if st.button("💾 Guardar metas"):
    df = goals_edit.copy()
    # Normalize target_date to string
    if not df.empty and "target_date" in df.columns:
        try:
            df["target_date"] = pd.to_datetime(df["target_date"]).dt.date.astype(str)
        except Exception:
            pass
    write_df_to_ws(ws_goals, df[goals_cols])
    st.success("Metas guardadas.")

# ----------------------------------
# KPIs / Gráficas / Forecast
# ----------------------------------
st.markdown("---")
st.subheader("📌 KPIs y gráficos")
totals = build_totals(income_df, expense_df, tx_df)
year_opts = sorted(totals["year"].unique())
year_kpi = st.selectbox("Año", year_opts, index=len(year_opts)-1)
dfy = totals[totals["year"]==year_kpi].sort_values("month")
dfy["inc_ma3"] = dfy["income"].rolling(3).mean()
dfy["exp_ma3"] = dfy["expense"].rolling(3).mean()
dfy["net_ma3"] = dfy["net"].rolling(3).mean()

c1,c2,c3,c4 = st.columns(4)
c1.metric(f"Ingresos YTD ({base_currency})", f"{dfy['income'].sum():,.0f}")
c2.metric(f"Gastos YTD ({base_currency})", f"{dfy['expense'].sum():,.0f}")
c3.metric(f"Ahorro Neto YTD ({base_currency})", f"{dfy['net'].sum():,.0f}")
srate = (1 - (dfy['expense'].sum()/dfy['income'].sum()))*100 if dfy['income'].sum()>0 else np.nan
c4.metric("Tasa de Ahorro", f"{srate:,.1f}%")

line = px.line(dfy, x="month", y=["income","expense","net","net_ma3"], markers=True, labels={"month":"Mes","value":f"Monto ({base_currency})"})
st.plotly_chart(line, use_container_width=True)

# Forecast neto (6 meses)
ser = dfy.set_index("month")["net"]
def forecast_series(series, periods=6):
    s = series.astype(float).dropna()
    if len(s) < 6:
        return None
    try:
        model = ExponentialSmoothing(s, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        fcast = fit.forecast(periods)
        return fcast
    except Exception:
        return None
fcast = forecast_series(ser, 6)
if fcast is not None:
    fc_df = pd.DataFrame({"month": list(ser.index) + list(range(int(ser.index.max())+1, int(ser.index.max())+1+len(fcast))),
                          "net": list(ser.values) + list(fcast.values),
                          "tipo": ["hist"]*len(ser) + ["forecast"]*len(fcast)})
    area = px.area(fc_df, x="month", y="net", color="tipo", title=f"Cash flow neto — histórico y forecast (6 meses) [{base_currency}]")
    st.plotly_chart(area, use_container_width=True)

# ----------------------------------
# Budget vs Actual (editor + semáforos)
# ----------------------------------
st.markdown("---")
st.subheader("📒 Budget vs Actual")

# Editor de budget por año
def budget_to_wide(df, year, kind):
    d = df[(df["year"].astype(str)==str(year)) & (df["type"]==kind)].copy()
    if d.empty:
        return pd.DataFrame(columns=["name"]+MONTHS)
    d["month"] = pd.to_numeric(d["month"], errors="coerce")
    w = d.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    w.columns = [REV_MONTHS.get(c,c) for c in w.columns]
    w = w.reset_index().rename(columns={"category":"name"})
    for m in MONTHS:
        if m not in w.columns:
            w[m] = 0.0
    return w[["name"]+MONTHS]

b_inc_w = budget_to_wide(budget_df, year_sel, "Ingreso")
b_exp_w = budget_to_wide(budget_df, year_sel, "Gasto")

colb1, colb2 = st.columns(2)
with colb1:
    st.markdown("**Budget de Ingresos**")
    b_inc_edit = st.data_editor(b_inc_w, num_rows="dynamic", use_container_width=True, key=f"binc_{year_sel}")
with colb2:
    st.markdown("**Budget de Gastos**")
    b_exp_edit = st.data_editor(b_exp_w, num_rows="dynamic", use_container_width=True, key=f"bexp_{year_sel}")

def wide_budget_to_tidy(wide, year, kind):
    rows = []
    for _, r in wide.iterrows():
        for m in MONTHS:
            v = r.get(m, 0.0)
            if pd.notna(v) and float(v) != 0.0:
                rows.append({"year": str(year), "month": MONTHS_MAP[m], "category": r["name"], "type": kind, "amount": float(v)})
    return pd.DataFrame(rows, columns=budget_cols)

if st.button("💾 Guardar budget del año"):
    b_inc_new = wide_budget_to_tidy(b_inc_edit, year_sel, "Ingreso")
    b_exp_new = wide_budget_to_tidy(b_exp_edit, year_sel, "Gasto")
    b2 = budget_df[budget_df["year"].astype(str)!=str(year_sel)].copy()
    b2 = pd.concat([b2, b_inc_new, b_exp_new], ignore_index=True)
    write_df_to_ws(ws_budget, b2[budget_cols])
    budget_df = b2
    st.success("Budget guardado.")

# Cálculo variances (totales y por categoría de gasto)
actual_tot = build_totals(income_df, expense_df, tx_df)
actual_tot_y = actual_tot[actual_tot["year"]==str(year_sel)][["month","income","expense","net"]].copy()

def sum_budget(df, year, kind):
    d = df[(df["year"].astype(str)==str(year)) & (df["type"]==kind)].groupby("month", as_index=False)["amount"].sum()
    d["month"] = pd.to_numeric(d["month"])
    return d.rename(columns={"amount": f"budget_{kind.lower()}"})
b_inc_tot = sum_budget(budget_df, year_sel, "Ingreso")
b_exp_tot = sum_budget(budget_df, year_sel, "Gasto")

variance = pd.DataFrame({"month": range(1,13)}).merge(actual_tot_y, on="month", how="left").fillna(0)
variance = variance.merge(b_inc_tot, on="month", how="left").fillna({"budget_ingreso":0})
variance = variance.merge(b_exp_tot, on="month", how="left").fillna({"budget_gasto":0})
variance["var_inc"] = variance["income"] - variance["budget_ingreso"]
variance["var_exp"] = variance["expense"] - variance["budget_gasto"]
variance["var_exp_pct"] = np.where(variance["budget_gasto"]>0, (variance["expense"]/variance["budget_gasto"] - 1), np.nan)

# Semáforos por % desvío en gasto
def color_for(pct):
    if pd.isna(pct):
        return "gray"
    if abs(pct) <= variance_green:
        return "green"
    if abs(pct) <= variance_yellow:
        return "orange"
    return "red"

variance["status"] = variance["var_exp_pct"].apply(color_for)

st.markdown("**Resumen mensual — Budget vs Actual (Gastos)**")
fig_var = px.bar(variance, x="month", y=["budget_gasto","expense"], barmode="group",
                 labels={"value":f"Monto ({base_currency})","month":"Mes","variable":""})
st.plotly_chart(fig_var, use_container_width=True)

# Tabla con colores
def style_row(row):
    color = row["status"]
    return [f"background-color: {color}; color: white" if col=="status" else "" for col in row.index]

styled = variance.copy()
styled["status"] = styled["status"].str.upper()
st.dataframe(styled, use_container_width=True)

# ----------------------------------
# Metas: progreso y proyección simple
# ----------------------------------
st.markdown("---")
st.subheader("🎯 Progreso de metas")
if not goals_df.empty:
    avg_net = dfy["net"].mean() if len(dfy)>0 else 0
    for _, g in goals_df.iterrows():
        name = g.get("name","Meta")
        target = float(g.get("target_amount",0) or 0)
        monthly = float(g.get("monthly_contrib",0) or 0)
        target_date = g.get("target_date","")
        # Progreso estimado: aporte mensual declarado
        months_left = None
        try:
            if target_date:
                td = pd.to_datetime(target_date)
                months_left = max(0, (td.year - date.today().year)*12 + (td.month - date.today().month))
        except Exception:
            pass
        projected = monthly * (months_left if months_left is not None else 12)
        pct = min(100.0, (projected/target*100) if target>0 else 0)
        st.write(f"**{name}** — objetivo: {target:,.0f} {base_currency} | aporte mensual: {monthly:,.0f} | fecha objetivo: {target_date or '-'}")
        st.progress(pct/100)
else:
    st.info("Agregá metas en la tabla GOALS (arriba) para ver el progreso.")

# ----------------------------------
# Exportación a Excel (backup completo)
# ----------------------------------
st.markdown("---")
st.subheader("⬇️ Exportación a Excel (backup completo)")

def build_excel_bytes():
    out = pd.ExcelWriter("presupuesto_export.xlsx", engine="openpyxl")
    # Raw tabs
    settings_df.to_excel(out, sheet_name="SETTINGS", index=False)
    income_df.to_excel(out, sheet_name="INCOME_CATS", index=False)
    expense_df.to_excel(out, sheet_name="EXPENSE_CATS", index=False)
    tx_df.to_excel(out, sheet_name="TRANSACTIONS", index=False)
    budget_df.to_excel(out, sheet_name="BUDGET", index=False)
    goals_df.to_excel(out, sheet_name="GOALS", index=False)
    # Aggregates
    totals.to_excel(out, sheet_name=f"TOTALS_{year_kpi}", index=False)
    variance.to_excel(out, sheet_name=f"VARIANCE_{year_sel}", index=False)
    out.close()
    with open("presupuesto_export.xlsx","rb") as f:
        return f.read()

if st.button("Generar backup Excel"):
    data = build_excel_bytes()
    st.download_button("Descargar backup Excel", data, file_name="presupuesto_backup.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("Persistencia activa con Google Sheets. Ahora con Budget vs Actual, Metas y Exportación a Excel.")
