
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Presupuesto Familiar — Pro", layout="wide")

st.title("📊 Presupuesto Familiar — Pro")
st.caption("Incluye editor tipo Excel, análisis de ingresos, simulación de escenarios, proyección de ahorro, multi-cuenta/moneda (UYU y USD) y tags.")

# -----------------------------
# Helpers
# -----------------------------
MONTHS = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
MONTHS_MAP = {m:i+1 for i,m in enumerate(MONTHS)}
REV_MONTHS_MAP = {v:k for k,v in MONTHS_MAP.items()}

def parse_year_sheet(df):
    df = df.copy()
    month_row_idx = None
    for i in range(len(df)):
        row_vals = df.iloc[i].astype(str).tolist()
        if sum(m in row_vals for m in MONTHS) >= 6:
            month_row_idx = i
            break
    if month_row_idx is None:
        return None

    col_map = {}
    for m in MONTHS:
        for j,val in enumerate(df.iloc[month_row_idx].astype(str)):
            if val == m:
                col_map[m] = j
                break
    if len(col_map) < 6:
        return None

    label_col = None
    for j in range(df.shape[1]):
        if df.iloc[:, j].astype(str).str.contains('GASTOS|INGRESOS|TIPO DE INGRESOS', case=False, regex=True).sum() >= 2:
            label_col = j
            break
    if label_col is None:
        label_col = 1

    idx_ing_header = df.index[df.iloc[:,label_col].astype(str).str.contains('TIPO DE INGRESOS', na=False)].tolist()
    idx_ing_total  = df.index[df.iloc[:,label_col].astype(str).str.contains('TOTAL DE INGRESOS', na=False)].tolist()
    idx_gas_header = df.index[df.iloc[:,label_col].astype(str).str.match('GASTOS$', na=False)].tolist()
    idx_gas_total  = df.index[df.iloc[:,label_col].astype(str).str.contains('TOTAL DE GASTOS', na=False)].tolist()

    def extract_rows(start_idx, end_idx):
        rows = []
        for i in range(start_idx+1, end_idx):
            name = df.iloc[i, label_col]
            if pd.isna(name): 
                continue
            data = {}
            for m, col in col_map.items():
                val = df.iloc[i, col]
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = np.nan
                data[m] = val
            rows.append({"name": str(name), **data})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    result = {"income_categories": pd.DataFrame(), "expense_categories": pd.DataFrame()}

    if idx_ing_header and idx_ing_total:
        inc_df = extract_rows(idx_ing_header[0], idx_ing_total[0])
        result["income_categories"] = inc_df

    if idx_gas_header and idx_gas_total:
        exp_df = extract_rows(idx_gas_header[0], idx_gas_total[0])
        result["expense_categories"] = exp_df

    def extract_total(idx_total):
        data = {}
        for m, col in col_map.items():
            val = df.iloc[idx_total, col]
            try:
                data[m] = float(val)
            except (TypeError, ValueError):
                data[m] = np.nan
        return pd.Series(data)

    totals = {}
    if idx_ing_total:
        totals['income']  = extract_total(idx_ing_total[0])
    if idx_gas_total:
        totals['expense'] = extract_total(idx_gas_total[0])
    result['totals'] = totals
    return result

def tidy_from_parsed(parsed):
    rows = []
    cat_exp_rows = []
    cat_inc_rows = []
    for year, content in parsed.items():
        for kind in ['income','expense']:
            s = content['totals'].get(kind, pd.Series())
            for m, val in s.items():
                if m in MONTHS_MAP:
                    rows.append({"year": int(year), "month": MONTHS_MAP[m], "kind": kind, "amount": val})
        exp_cat = content.get("expense_categories", pd.DataFrame())
        if not exp_cat.empty:
            for _, r in exp_cat.iterrows():
                for m, mon_num in MONTHS_MAP.items():
                    if m in exp_cat.columns:
                        val = r.get(m, np.nan)
                        if pd.notna(val):
                            cat_exp_rows.append({
                                "year": int(year), "month": mon_num, "category": r["name"], "amount": float(val),
                                "account": "Excel", "currency": "UYU", "tags": ""
                            })
        inc_cat = content.get("income_categories", pd.DataFrame())
        if not inc_cat.empty:
            for _, r in inc_cat.iterrows():
                for m, mon_num in MONTHS_MAP.items():
                    if m in inc_cat.columns:
                        val = r.get(m, np.nan)
                        if pd.notna(val):
                            cat_inc_rows.append({
                                "year": int(year), "month": mon_num, "category": r["name"], "amount": float(val),
                                "account": "Excel", "currency": "UYU", "tags": ""
                            })
    totals_df = pd.DataFrame(rows).dropna()
    by_exp_cat = pd.DataFrame(cat_exp_rows)
    by_inc_cat = pd.DataFrame(cat_inc_rows)
    return totals_df, by_exp_cat, by_inc_cat

def load_workbook(file_bytes=None):
    if file_bytes is None:
        return None, None, None, None
    xls = pd.ExcelFile(BytesIO(file_bytes))
    sheets = {}
    for name in xls.sheet_names:
        df = xls.parse(name, header=None)
        parsed = parse_year_sheet(df)
        if parsed: sheets[name] = parsed
    totals_df, exp_by_cat, inc_by_cat = tidy_from_parsed(sheets)
    return sheets, totals_df, exp_by_cat, inc_by_cat

def forecast_series(series, periods=6, seasonal=None):
    s = series.astype(float).dropna()
    if len(s) < 6:
        return None
    try:
        model = ExponentialSmoothing(s, trend="add", seasonal=("add" if seasonal else None), seasonal_periods=(12 if seasonal else None), initialization_method="estimated")
        fit = model.fit()
        fcast = fit.forecast(periods)
        return fcast
    except Exception:
        return None

def compound_growth(monthly_contrib, annual_rate, months):
    if annual_rate is None or annual_rate == 0:
        return monthly_contrib * months
    r = annual_rate / 12.0 / 100.0
    # Future value of annuity due (payments at period end assumed here)
    fv = monthly_contrib * (((1 + r)**months - 1) / r)
    return fv

# -----------------------------
# Sidebar: inputs globales
# -----------------------------
with st.sidebar:
    st.header("📥 Datos de entrada")
    uploaded = st.file_uploader("Subí el Excel (2023/2024/2025)", type=["xls","xlsx"])

    st.markdown("---")
    st.subheader("💱 Moneda y cuentas")
    base_currency = st.selectbox("Moneda base", ["UYU","USD"], index=0)
    fx = st.number_input("Tipo de cambio (USD → UYU)", min_value=1.0, value=40.0, step=0.5)
    st.caption("Los datos del Excel se asumen en UYU. Las transacciones manuales permiten UYU o USD.")

    st.markdown("---")
    st.subheader("🎯 Objetivos & Alertas")
    target_savings = st.number_input("Objetivo de ahorro mensual (en moneda base)", min_value=0.0, value=0.0, step=100.0)
    alert_threshold = st.slider("Alerta cuando el gasto mensual exceda su media 3m por (%)", 5, 100, 25)
    compare_prev_year = st.toggle("Comparar contra mismo mes del año anterior", True)

    st.markdown("---")
    st.subheader("🧪 Simulación (What-if)")
    sim_inc = st.slider("Ajuste ingresos (% mensual)", -50, 50, 0)
    sim_exp = st.slider("Ajuste gastos (% mensual)", -50, 50, 0)
    st.caption("Aplica un delta uniforme sobre el año seleccionado para estimar impacto.")

# -----------------------------
# Load workbook
# -----------------------------
if "transactions" not in st.session_state:
    st.session_state["transactions"] = pd.DataFrame(columns=[
        "date","type","category","amount","account","currency","tags","notes"
    ])

sheets, totals_df, exp_by_cat, inc_by_cat = (None, None, None, None)
if uploaded is not None:
    sheets, totals_df, exp_by_cat, inc_by_cat = load_workbook(uploaded.read())

if totals_df is None or totals_df.empty:
    st.info("👋 Cargá tu Excel para continuar.")
    st.stop()

# -----------------------------
# Editor tipo Excel
# -----------------------------
years = sorted({int(y) for y in totals_df["year"].unique()})
st.subheader("🧾 Editor tipo Excel — por año")
year_tab = st.selectbox("Año a editar", years, index=len(years)-1)

def get_wide(df, year):
    if df is None or df.empty:
        return pd.DataFrame(columns=["name"] + MONTHS)
    dfy = df[df["year"]==year]
    wide = dfy.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    if wide.empty:
        return pd.DataFrame(columns=["name"] + MONTHS)
    wide.columns = [REV_MONTHS_MAP.get(c, c) for c in wide.columns]
    wide = wide.reset_index().rename(columns={"category":"name"})
    for m in MONTHS:
        if m not in wide.columns:
            wide[m] = 0.0
    return wide[["name"] + MONTHS]

inc_wide = get_wide(inc_by_cat, year_tab)
exp_wide = get_wide(exp_by_cat, year_tab)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Ingresos (editor)**")
    inc_edit = st.data_editor(
        inc_wide,
        num_rows="dynamic",
        use_container_width=True,
        column_config={m: st.column_config.NumberColumn(m, format="%.2f", step=1) for m in MONTHS} | {
            "name": st.column_config.TextColumn("Categoría")
        },
        key=f"inc_editor_{year_tab}"
    )
with col2:
    st.markdown("**Gastos (editor)**")
    exp_edit = st.data_editor(
        exp_wide,
        num_rows="dynamic",
        use_container_width=True,
        column_config={m: st.column_config.NumberColumn(m, format="%.2f", step=1) for m in MONTHS} | {
            "name": st.column_config.TextColumn("Categoría")
        },
        key=f"exp_editor_{year_tab}"
    )

def wide_to_tidy(wide, year, kind):
    rows = []
    for _, r in wide.iterrows():
        for m in MONTHS:
            v = r.get(m, 0.0)
            if pd.notna(v) and float(v) != 0.0:
                rows.append({
                    "year": int(year), "month": MONTHS_MAP[m],
                    "category": r["name"], "amount": float(v),
                    "account": "Excel-Editado", "currency": "UYU", "tags": ""
                })
    return pd.DataFrame(rows)

inc_tidy = wide_to_tidy(inc_edit, year_tab, "income")
exp_tidy = wide_to_tidy(exp_edit, year_tab, "expense")

# Reemplazar datos del año por los editados
inc_by_cat2 = inc_by_cat[inc_by_cat["year"]!=year_tab].copy() if inc_by_cat is not None else pd.DataFrame()
exp_by_cat2 = exp_by_cat[exp_by_cat["year"]!=year_tab].copy() if exp_by_cat is not None else pd.DataFrame()
inc_by_cat2 = pd.concat([inc_by_cat2, inc_tidy], ignore_index=True)
exp_by_cat2 = pd.concat([exp_by_cat2, exp_tidy], ignore_index=True)

# -----------------------------
# Transacciones manuales (multi-cuenta/moneda + tags)
# -----------------------------
st.subheader("🧾 Transacciones manuales (multi-cuenta / UYU-USD / tags)")
tx = st.session_state["transactions"]
tx_edit = st.data_editor(
    tx,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "date": st.column_config.DateColumn("Fecha"),
        "type": st.column_config.SelectboxColumn("Tipo", options=["Ingreso","Gasto"]),
        "category": st.column_config.TextColumn("Categoría"),
        "amount": st.column_config.NumberColumn("Monto", step=1, format="%.2f"),
        "account": st.column_config.TextColumn("Cuenta"),
        "currency": st.column_config.SelectboxColumn("Moneda", options=["UYU","USD"]),
        "tags": st.column_config.TextColumn("Tags (coma-separados)"),
        "notes": st.column_config.TextColumn("Notas"),
    },
    key="tx_editor"
)
st.session_state["transactions"] = tx_edit

# Filtros avanzados
st.markdown("#### 🔎 Filtros")
colf1, colf2, colf3 = st.columns(3)
with colf1:
    account_filter = st.text_input("Cuenta (contiene)", "")
with colf2:
    tag_filter = st.text_input("Tags (contiene)", "")
with colf3:
    show_currency = st.selectbox("Ver montos en", [base_currency, ("USD" if base_currency=="UYU" else "UYU")], index=0)

def convert_amount(row_amount, row_cur, to_cur, fx):
    if row_cur == to_cur:
        return row_amount
    # USD <-> UYU only
    if row_cur == "USD" and to_cur == "UYU":
        return row_amount * fx
    if row_cur == "UYU" and to_cur == "USD":
        return row_amount / fx
    return row_amount

# Tidy total a partir de categorias + transacciones
def build_totals(inc_by_cat, exp_by_cat, tx_df):
    # Manual tx tidy (monthly)
    manual = pd.DataFrame()
    if tx_df is not None and not tx_df.empty:
        t = tx_df.dropna(subset=["date","amount"]).copy()
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t = t.dropna(subset=["date"])
        if account_filter:
            t = t[t["account"].fillna("").str.contains(account_filter, case=False, na=False)]
        if tag_filter:
            t = t[t["tags"].fillna("").str.contains(tag_filter, case=False, na=False)]
        t["year"] = t["date"].dt.year
        t["month"] = t["date"].dt.month
        sign = np.where(t["type"]=="Ingreso", 1, -1)
        # Convert amounts to base currency for aggregation
        t["amount_base"] = [convert_amount(a, c, base_currency, fx) for a,c in zip(t["amount"], t["currency"])]
        manual = t.groupby(["year","month","type"], as_index=False)["amount_base"].sum()

    # Build from categories (assume UYU → convert if needed for display)
    inc = inc_by_cat.groupby(["year","month"], as_index=False)["amount"].sum().rename(columns={"amount":"income"})
    exp = exp_by_cat.groupby(["year","month"], as_index=False)["amount"].sum().rename(columns={"amount":"expense"})
    merged = pd.merge(inc, exp, on=["year","month"], how="outer").fillna(0)

    # Apply manual adjustments
    if not manual.empty:
        for _, r in manual.iterrows():
            mask = (merged["year"]==r["year"]) & (merged["month"]==r["month"])
            if not mask.any():
                merged = pd.concat([merged, pd.DataFrame({"year":[r["year"]], "month":[r["month"]], "income":[0.0], "expense":[0.0]})], ignore_index=True)
                mask = (merged["year"]==r["year"]) & (merged["month"]==r["month"])
            if r["type"]=="Ingreso":
                merged.loc[mask, "income"] = merged.loc[mask, "income"] + r["amount_base"]
            else:
                merged.loc[mask, "expense"] = merged.loc[mask, "expense"] + r["amount_base"]

    # Currency conversion for display
    if show_currency != "UYU":
        # convert from base to other display currency
        if base_currency == "UYU" and show_currency == "USD":
            merged["income"] = merged["income"] / fx
            merged["expense"] = merged["expense"] / fx
        elif base_currency == "USD" and show_currency == "UYU":
            merged["income"] = merged["income"] * fx
            merged["expense"] = merged["expense"] * fx

    merged["net"] = merged["income"] - merged["expense"]
    return merged

totals_df2 = build_totals(inc_by_cat2, exp_by_cat2, st.session_state["transactions"])

# -----------------------------
# KPIs y análisis (incluye dependencia de ingresos)
# -----------------------------
st.markdown("---")
st.subheader("📌 KPIs y análisis")
year_sel = st.selectbox("Año", sorted(totals_df2["year"].unique()), index=len(sorted(totals_df2["year"].unique()))-1)

df_year = totals_df2[totals_df2["year"]==year_sel].sort_values("month").copy()
df_year["inc_ma3"] = df_year["income"].rolling(3).mean()
df_year["exp_ma3"] = df_year["expense"].rolling(3).mean()
df_year["net_ma3"] = df_year["net"].rolling(3).mean()

ytd_income  = df_year["income"].sum()
ytd_expense = df_year["expense"].sum()
ytd_net     = df_year["net"].sum()
savings_rate = (1 - (ytd_expense / ytd_income)) * 100 if ytd_income else np.nan

k1,k2,k3,k4 = st.columns(4)
k1.metric(f"Ingresos YTD ({show_currency})", f"{ytd_income:,.0f}")
k2.metric(f"Gastos YTD ({show_currency})", f"{ytd_expense:,.0f}")
k3.metric(f"Ahorro Neto YTD ({show_currency})", f"{ytd_net:,.0f}")
k4.metric("Tasa de Ahorro", f"{savings_rate:,.1f}%")

# Dependencia de ingresos (por fuente/categoría)
st.markdown("#### 📈 Dependencia de ingresos por fuente")
inc_src = inc_by_cat2[inc_by_cat2["year"]==year_sel].groupby("category", as_index=False)["amount"].sum()
inc_total_base = inc_src["amount"].sum()
if inc_total_base > 0:
    inc_src["share_%"] = (inc_src["amount"] / inc_total_base) * 100
    pie_inc = px.pie(inc_src.sort_values("share_%", ascending=False), names="category", values="share_%", title="Participación de cada fuente de ingreso (base UYU)")
    st.plotly_chart(pie_inc, use_container_width=True)
    top_dep = inc_src.sort_values("share_%", ascending=False).head(3)
    st.caption("Top dependencia: " + ", ".join([f"{r.category}: {r['share_%']:.1f}%" for _,r in top_dep.iterrows()]))
else:
    st.info("No hay datos de ingresos por fuente para este año.")

# -----------------------------
# Gráficos principales
# -----------------------------
st.markdown("### 📉 Evolución mensual")
line = px.line(df_year, x="month", y=["income","expense","net","net_ma3"], markers=True,
               labels={"value":f"Monto ({show_currency})","month":"Mes","variable":"Serie"})
st.plotly_chart(line, use_container_width=True)

st.markdown("### 🔥 Heatmap de gastos por categoría")
cats_year = exp_by_cat2[exp_by_cat2["year"]==year_sel]
if not cats_year.empty:
    # Filtros por account/tag sobre categorías via transacciones no aplican; aquí son datos Excel/edición
    heat = cats_year.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    heat = heat[sorted(heat.columns)]
    im = px.imshow(heat, aspect="auto", labels=dict(x="Mes", y="Categoría", color="Gasto (UYU)"))
    st.plotly_chart(im, use_container_width=True)
else:
    st.info("No hay desglose de categorías para gastos en este año.")

# -----------------------------
# Simulación de escenarios (what-if)
# -----------------------------
st.markdown("---")
st.subheader("🧪 Simulación de escenarios")

sim_df = df_year.copy()
sim_df["income_sim"]  = sim_df["income"] * (1 + sim_inc/100.0)
sim_df["expense_sim"] = sim_df["expense"] * (1 + sim_exp/100.0)
sim_df["net_sim"]     = sim_df["income_sim"] - sim_df["expense_sim"]

c1,c2,c3 = st.columns(3)
c1.metric("Δ Ingresos (anual)", f"{(sim_df['income_sim'].sum()-df_year['income'].sum()):,.0f} {show_currency}")
c2.metric("Δ Gastos (anual)", f"{(sim_df['expense_sim'].sum()-df_year['expense'].sum()):,.0f} {show_currency}")
c3.metric("Δ Neto (anual)", f"{(sim_df['net_sim'].sum()-df_year['net'].sum()):,.0f} {show_currency}")

sim_fig = px.line(sim_df, x="month", y=["net","net_sim"], markers=True, title="Neto vs Neto simulado")
st.plotly_chart(sim_fig, use_container_width=True)

# -----------------------------
# Forecast y proyección de ahorro a largo plazo
# -----------------------------
st.markdown("---")
st.subheader("🔮 Forecast + Proyección de ahorro")
multi_year = len(totals_df2["year"].unique()) >= 2
ser_net = df_year.set_index("month")["net"]
f_net = None
try:
    f_net = forecast_series(ser_net, 6, seasonal=multi_year)
except Exception:
    f_net = None

if f_net is not None:
    fc_df = pd.DataFrame({"month": list(ser_net.index) + list(range(int(ser_net.index.max())+1, int(ser_net.index.max())+1+len(f_net))),
                          "net": list(ser_net.values) + list(f_net.values),
                          "tipo": ["hist"]*len(ser_net) + ["forecast"]*len(f_net)})
    area = px.area(fc_df, x="month", y="net", color="tipo", title=f"Cash flow neto — histórico y forecast (6 meses) [{show_currency}]")
    st.plotly_chart(area, use_container_width=True)
    st.caption(f"Proyección 6m: {f_net.sum():,.0f} {show_currency} | Promedio mensual proyectado: {f_net.mean():,.0f} {show_currency}")
else:
    st.info("Se necesitan al menos 6 puntos válidos para el forecast de neto.")

st.markdown("#### 💡 Proyección de ahorro a 1, 3 y 5 años")
colp1, colp2 = st.columns(2)
with colp1:
    contrib = st.number_input(f"Aporte mensual estimado ({show_currency})", min_value=0.0, value=float(max(0, ser_net.mean() if len(ser_net)>0 else 0)), step=100.0)
with colp2:
    rate = st.number_input("Tasa anual estimada (%)", min_value=0.0, value=3.0, step=0.5)

proj = {
    "1 año": compound_growth(contrib, rate, 12),
    "3 años": compound_growth(contrib, rate, 36),
    "5 años": compound_growth(contrib, rate, 60),
}
pcol1,pcol2,pcol3 = st.columns(3)
pcol1.metric("1 año", f"{proj['1 año']:,.0f} {show_currency}")
pcol2.metric("3 años", f"{proj['3 años']:,.0f} {show_currency}")
pcol3.metric("5 años", f"{proj['5 años']:,.0f} {show_currency}")

st.caption("La proyección usa interés compuesto simple mensual con aportes constantes. No incluye impuestos ni variaciones de tasa.")

# -----------------------------
# Alertas (reutiliza configuración previa)
# -----------------------------
st.markdown("---")
st.subheader("🚨 Alertas")
alerts = []

# Neto vs objetivo
if target_savings and target_savings > 0 and not df_year.empty:
    for _, r in df_year.iterrows():
        if r["net"] < target_savings:
            alerts.append(f"Mes {REV_MONTHS_MAP[int(r['month'])]}: neto {r['net']:,.0f} por debajo del objetivo ({target_savings:,.0f}).")

# Gasto por categoría vs MA3
if not cats_year.empty:
    for m in sorted(cats_year["month"].unique()):
        grp_m = cats_year[cats_year["month"]==m]
        for cat in grp_m["category"].unique():
            series = cats_year[cats_year["category"]==cat].set_index("month").sort_index()["amount"]
            ma3 = series.rolling(3).mean().get(m, np.nan)
            val = series.get(m, np.nan)
            if pd.notna(val) and pd.notna(ma3) and ma3 > 0:
                pct = (val/ma3 - 1)*100
                if pct >= alert_threshold:
                    alerts.append(f"Mes {REV_MONTHS_MAP[m]}: **{cat}** {pct:.0f}% sobre su media móvil 3m ({val:,.0f} UYU vs {ma3:,.0f} UYU).")

# Comparación con año previo
if compare_prev_year and (year_sel-1 in years):
    prev = totals_df2[totals_df2["year"]==year_sel-1].sort_values("month")
    merged = df_year[["month","expense"]].merge(prev[["month","expense"]].rename(columns={"expense":"expense_prev"}), on="month", how="left")
    for _, r in merged.dropna().iterrows():
        if r["expense_prev"] > 0:
            pct = (r["expense"]/r["expense_prev"] - 1)*100
            if pct >= alert_threshold:
                alerts.append(f"Mes {REV_MONTHS_MAP[int(r['month'])]}: gasto {pct:.0f}% mayor que el mismo mes del {year_sel-1}.")

if alerts:
    for a in alerts[:8]:
        st.warning(a)
    if len(alerts) > 8:
        st.info(f"…y {len(alerts)-8} alertas más (ajustá el umbral en la barra lateral).")
else:
    st.success("Sin alertas con los parámetros actuales.")

st.markdown("---")
st.caption("Tip: Para persistir transacciones manuales, exportá CSV o conectá una Google Sheet. La conversión de moneda es USD↔︎UYU con el tipo de cambio ingresado.")
