
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, date
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Presupuesto Familiar — Excel Plus", layout="wide")
st.title("📊 Presupuesto Familiar — Excel (Plus)")
st.caption("Carga tu Excel y usa editor tipo Excel, Budget vs Actual, simulación, proyección de ahorro, multi-moneda (UYU↔USD), tags/filtros y exportación.")

# ----------------------------------
# Helpers para parsear tu Excel
# ----------------------------------
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
                try: val = float(val)
                except (TypeError, ValueError): val = np.nan
                data[m] = val
            rows.append({"name": str(name), **data})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    result = {"income_categories": pd.DataFrame(), "expense_categories": pd.DataFrame()}
    if idx_ing_header and idx_ing_total:
        result["income_categories"] = extract_rows(idx_ing_header[0], idx_ing_total[0])
    if idx_gas_header and idx_gas_total:
        result["expense_categories"] = extract_rows(idx_gas_header[0], idx_gas_total[0])

    def extract_total(idx_total):
        data = {}
        for m, col in col_map.items():
            val = df.iloc[idx_total, col]
            try: data[m] = float(val)
            except (TypeError, ValueError): data[m] = np.nan
        return pd.Series(data)

    totals = {}
    if idx_ing_total: totals['income']  = extract_total(idx_ing_total[0])
    if idx_gas_total: totals['expense'] = extract_total(idx_gas_total[0])
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
                            cat_exp_rows.append({"year": int(year), "month": mon_num, "category": r["name"], "amount": float(val)})
        inc_cat = content.get("income_categories", pd.DataFrame())
        if not inc_cat.empty:
            for _, r in inc_cat.iterrows():
                for m, mon_num in MONTHS_MAP.items():
                    if m in inc_cat.columns:
                        val = r.get(m, np.nan)
                        if pd.notna(val):
                            cat_inc_rows.append({"year": int(year), "month": mon_num, "category": r["name"], "amount": float(val)})
    totals_df = pd.DataFrame(rows).dropna()
    by_exp_cat = pd.DataFrame(cat_exp_rows)
    by_inc_cat = pd.DataFrame(cat_inc_rows)
    return totals_df, by_exp_cat, by_inc_cat

def load_workbook(file_bytes):
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
    if len(s) < 6: return None
    try:
        model = ExponentialSmoothing(s, trend="add", seasonal=("add" if seasonal else None), seasonal_periods=(12 if seasonal else None), initialization_method="estimated")
        fit = model.fit()
        return fit.forecast(periods)
    except Exception:
        return None

# ----------------------------------
# Sidebar: configuración general
# ----------------------------------
with st.sidebar:
    st.header("📥 Datos")
    uploaded = st.file_uploader("Subí el Excel (2023/2024/2025)", type=["xls","xlsx"])
    st.markdown("---")
    st.subheader("💱 Moneda y filtros")
    base_currency = st.selectbox("Moneda base", ["UYU","USD"], index=0)
    fx = st.number_input("Tipo de cambio (USD→UYU)", min_value=1.0, value=40.0, step=0.5)
    st.caption("Los datos del Excel se asumen en UYU. Las transacciones manuales pueden ser UYU o USD.")
    st.markdown("---")
    st.subheader("🎯 Presupuesto & Alertas")
    target_savings = st.number_input("Objetivo de ahorro mensual (en moneda base)", min_value=0.0, value=0.0, step=100.0)
    alert_threshold = st.slider("Alerta: gasto mensual > media móvil 3m por (%)", 5, 100, 25)
    variance_green = st.number_input("Budget OK si desvío ≤", min_value=0.0, value=0.05, step=0.01, help="Ej. 0.05 = 5%")
    variance_yellow = st.number_input("Budget atención si desvío ≤", min_value=0.0, value=0.15, step=0.01, help="Rojo si supera esto.")
    st.markdown("---")
    st.subheader("🧪 Simulación (what‑if)")
    sim_inc = st.slider("Ajuste ingresos (%)", -50, 50, 0)
    sim_exp = st.slider("Ajuste gastos (%)", -50, 50, 0)
    st.caption("Aplica al año seleccionado.")

if uploaded is None:
    st.info("Cargá tu Excel para continuar.")
    st.stop()

sheets, totals_df, exp_by_cat, inc_by_cat = load_workbook(uploaded.read())
if totals_df is None or totals_df.empty:
    st.error("No pude leer totales del Excel. Verifica el formato.")
    st.stop()

# ----------------------------------
# Editor tipo Excel (categoría x mes) + transacciones manuales
# ----------------------------------
years = sorted({int(y) for y in totals_df["year"].unique()})
st.subheader("🧾 Editor tipo Excel — por año (ingresos/gastos)")
year_tab = st.selectbox("Año a editar", years, index=len(years)-1)

def to_wide(df, year):
    if df is None or df.empty:
        return pd.DataFrame(columns=["name"] + MONTHS)
    dfy = df[df["year"]==year]
    w = dfy.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    if w.empty:
        return pd.DataFrame(columns=["name"] + MONTHS)
    w.columns = [REV_MONTHS_MAP.get(c, c) for c in w.columns]
    w = w.reset_index().rename(columns={"category":"name"})
    for m in MONTHS:
        if m not in w.columns:
            w[m] = 0.0
    return w[["name"] + MONTHS]

inc_wide = to_wide(inc_by_cat, year_tab)
exp_wide = to_wide(exp_by_cat, year_tab)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Ingresos (editor)**")
    inc_edit = st.data_editor(
        inc_wide, num_rows="dynamic", use_container_width=True,
        column_config={m: st.column_config.NumberColumn(m, format="%.2f", step=1) for m in MONTHS} | {
            "name": st.column_config.TextColumn("Categoría")
        },
        key=f"inc_editor_{year_tab}"
    )
with col2:
    st.markdown("**Gastos (editor)**")
    exp_edit = st.data_editor(
        exp_wide, num_rows="dynamic", use_container_width=True,
        column_config={m: st.column_config.NumberColumn(m, format="%.2f", step=1) for m in MONTHS} | {
            "name": st.column_config.TextColumn("Categoría")
        },
        key=f"exp_editor_{year_tab}"
    )

def wide_to_tidy(wide, year):
    rows = []
    for _, r in wide.iterrows():
        for m in MONTHS:
            v = r.get(m, 0.0)
            if pd.notna(v) and float(v) != 0.0:
                rows.append({"year": int(year), "month": MONTHS_MAP[m], "category": r["name"], "amount": float(v)})
    return pd.DataFrame(rows)

inc_by_cat2 = inc_by_cat[inc_by_cat["year"]!=year_tab].copy() if inc_by_cat is not None else pd.DataFrame()
exp_by_cat2 = exp_by_cat[exp_by_cat["year"]!=year_tab].copy() if exp_by_cat is not None else pd.DataFrame()
inc_by_cat2 = pd.concat([inc_by_cat2, wide_to_tidy(inc_edit, year_tab)], ignore_index=True)
exp_by_cat2 = pd.concat([exp_by_cat2, wide_to_tidy(exp_edit, year_tab)], ignore_index=True)

# Transacciones manuales (multi‑moneda, cuenta, tags)
st.subheader("🧾 Transacciones manuales (opcional)")
if "transactions" not in st.session_state:
    st.session_state["transactions"] = pd.DataFrame(columns=["date","type","category","amount","account","currency","tags","notes"])
tx = st.session_state["transactions"]
tx_edit = st.data_editor(
    tx, num_rows="dynamic", use_container_width=True,
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

st.markdown("#### 🔎 Filtros (aplicados a transacciones manuales)")
colf1, colf2, colf3 = st.columns(3)
with colf1: account_filter = st.text_input("Cuenta (contiene)", "")
with colf2: tag_filter = st.text_input("Tags (contiene)", "")
with colf3: show_currency = st.selectbox("Ver montos en", [base_currency, ("USD" if base_currency=="UYU" else "UYU")], index=0)

def convert_amount(amount, cur_from, to_cur, fx):
    if cur_from == to_cur: return amount
    if cur_from == "USD" and to_cur == "UYU": return amount * fx
    if cur_from == "UYU" and to_cur == "USD": return amount / fx
    return amount

def build_totals(inc_by_cat, exp_by_cat, tx_df):
    inc = inc_by_cat.groupby(["year","month"], as_index=False)["amount"].sum().rename(columns={"amount":"income"})
    exp = exp_by_cat.groupby(["year","month"], as_index=False)["amount"].sum().rename(columns={"amount":"expense"})
    merged = pd.merge(inc, exp, on=["year","month"], how="outer").fillna(0)
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
        # Convert to base currency first
        t["amount_base"] = [convert_amount(a, c, base_currency, fx) for a,c in zip(t["amount"], t["currency"])]
        inc_tx = t[t["type"]=="Ingreso"].groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"inc_tx"})
        exp_tx = t[t["type"]=="Gasto"].groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"exp_tx"})
        merged = pd.merge(merged, inc_tx, on=["year","month"], how="left").fillna({"inc_tx":0})
        merged = pd.merge(merged, exp_tx, on=["year","month"], how="left").fillna({"exp_tx":0})
        merged["income"]  = merged["income"] + merged["inc_tx"]
        merged["expense"] = merged["expense"] + merged["exp_tx"]
    # Convert to display currency
    if show_currency != "UYU":
        if base_currency == "UYU" and show_currency == "USD":
            merged["income"] = merged["income"]/fx; merged["expense"] = merged["expense"]/fx
        elif base_currency == "USD" and show_currency == "UYU":
            merged["income"] = merged["income"]*fx; merged["expense"] = merged["expense"]*fx
    merged["net"] = merged["income"] - merged["expense"]
    return merged

totals_df2 = build_totals(inc_by_cat2, exp_by_cat2, st.session_state["transactions"])

# ----------------------------------
# KPIs, gráficos, análisis y simulación
# ----------------------------------
st.markdown("---")
st.subheader("📌 KPIs y análisis")
year_sel = st.selectbox("Año", sorted(totals_df2["year"].unique()), index=len(sorted(totals_df2["year"].unique()))-1)
df_year = totals_df2[totals_df2["year"]==year_sel].sort_values("month").copy()
df_year["inc_ma3"] = df_year["income"].rolling(3).mean()
df_year["exp_ma3"] = df_year["expense"].rolling(3).mean()
df_year["net_ma3"] = df_year["net"].rolling(3).mean()

k1,k2,k3,k4 = st.columns(4)
k1.metric(f"Ingresos YTD ({show_currency})", f"{df_year['income'].sum():,.0f}")
k2.metric(f"Gastos YTD ({show_currency})", f"{df_year['expense'].sum():,.0f}")
k3.metric(f"Ahorro Neto YTD ({show_currency})", f"{df_year['net'].sum():,.0f}")
srate = (1 - (df_year['expense'].sum()/df_year['income'].sum()))*100 if df_year['income'].sum()>0 else np.nan
k4.metric("Tasa de Ahorro", f"{srate:,.1f}%")

# Dependencia de ingresos por fuente
st.markdown("#### 📈 Dependencia de ingresos por fuente (base UYU)")
inc_src = inc_by_cat2[inc_by_cat2["year"]==year_sel].groupby("category", as_index=False)["amount"].sum()
if inc_src["amount"].sum() > 0:
    inc_src["share_%"] = (inc_src["amount"]/inc_src["amount"].sum())*100
    pie_inc = px.pie(inc_src.sort_values("share_%", ascending=False), names="category", values="share_%")
    st.plotly_chart(pie_inc, use_container_width=True)
else:
    st.info("No hay datos de ingresos por fuente para este año.")

st.markdown("### 📉 Evolución mensual")
line = px.line(df_year, x="month", y=["income","expense","net","net_ma3"], markers=True,
               labels={"value":f"Monto ({show_currency})","month":"Mes","variable":"Serie"})
st.plotly_chart(line, use_container_width=True)

# Heatmap gastos por categoría
cats_year = exp_by_cat2[exp_by_cat2["year"]==year_sel]
if not cats_year.empty:
    heat = cats_year.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    heat = heat[sorted(heat.columns)]
    im = px.imshow(heat, aspect="auto", labels=dict(x="Mes", y="Categoría", color="Gasto (UYU)"))
    st.markdown("### 🔥 Heatmap de gastos por categoría (UYU)")
    st.plotly_chart(im, use_container_width=True)

# Simulación
st.markdown("---")
st.subheader("🧪 Simulación de escenarios (what‑if)")
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

# Forecast + Proyección de ahorro
st.markdown("---")
st.subheader("🔮 Forecast + Proyección de ahorro")
multi_year = len(totals_df2["year"].unique()) >= 2
ser_net = df_year.set_index("month")["net"]
f_net = forecast_series(ser_net, 6, seasonal=multi_year)
if f_net is not None:
    fc_df = pd.DataFrame({"month": list(ser_net.index) + list(range(int(ser_net.index.max())+1, int(ser_net.index.max())+1+len(f_net))),
                          "net": list(ser_net.values) + list(f_net.values),
                          "tipo": ["hist"]*len(ser_net) + ["forecast"]*len(f_net)})
    area = px.area(fc_df, x="month", y="net", color="tipo", title=f"Cash flow neto — histórico y forecast (6 meses) [{show_currency}]")
    st.plotly_chart(area, use_container_width=True)
    st.caption(f"Proyección 6m: {f_net.sum():,.0f} {show_currency} | Promedio mensual proyectado: {f_net.mean():,.0f} {show_currency}")
else:
    st.info("Se necesitan al menos 6 puntos válidos para el forecast de neto.")

colp1, colp2 = st.columns(2)
with colp1:
    contrib = st.number_input(f"Aporte mensual estimado ({show_currency})", min_value=0.0, value=float(max(0, ser_net.mean() if len(ser_net)>0 else 0)), step=100.0)
with colp2:
    rate = st.number_input("Tasa anual estimada (%)", min_value=0.0, value=3.0, step=0.5)
def compound_growth(monthly_contrib, annual_rate, months):
    if annual_rate == 0: return monthly_contrib * months
    r = annual_rate / 12.0 / 100.0
    return monthly_contrib * (((1 + r)**months - 1) / r)
proj = {"1 año": compound_growth(contrib, rate, 12),
        "3 años": compound_growth(contrib, rate, 36),
        "5 años": compound_growth(contrib, rate, 60)}
pp1,pp2,pp3 = st.columns(3)
pp1.metric("1 año", f"{proj['1 año']:,.0f} {show_currency}")
pp2.metric("3 años", f"{proj['3 años']:,.0f} {show_currency}")
pp3.metric("5 años", f"{proj['5 años']:,.0f} {show_currency}")

# Alertas
st.markdown("---")
st.subheader("🚨 Alertas")
alerts = []
if target_savings and target_savings > 0 and not df_year.empty:
    for _, r in df_year.iterrows():
        if r["net"] < target_savings:
            alerts.append(f"Mes {REV_MONTHS_MAP[int(r['month'])]}: neto {r['net']:,.0f} por debajo del objetivo ({target_savings:,.0f}).")
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
if alerts:
    for a in alerts[:8]: st.warning(a)
    if len(alerts)>8: st.info(f"…y {len(alerts)-8} alertas más. Ajustá los umbrales.")
else:
    st.success("Sin alertas con los parámetros actuales.")

# ----------------------------------
# BUDGET vs ACTUAL
# ----------------------------------
st.markdown("---")
st.subheader("📒 Budget vs Actual (por categoría y mes)")
st.caption("Carga tu budget por categoría y mes. Se compara contra el real del Excel + transacciones.")

# Editor de budget wide para el año seleccionado
if "budget" not in st.session_state:
    # Estructura: year, month, category, type (Ingreso/Gasto), amount
    st.session_state["budget"] = pd.DataFrame(columns=["year","month","category","type","amount"])

def budget_to_wide(budget_df, year, kind):
    d = budget_df[(budget_df["year"]==year) & (budget_df["type"]==kind)].copy()
    if d.empty:
        return pd.DataFrame(columns=["name"]+MONTHS)
    w = d.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    w.columns = [REV_MONTHS_MAP.get(c,c) for c in w.columns]
    w = w.reset_index().rename(columns={"category":"name"})
    for m in MONTHS:
        if m not in w.columns: w[m] = 0.0
    return w[["name"]+MONTHS]

budget_df = st.session_state["budget"]
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
                rows.append({"year": int(year), "month": MONTHS_MAP[m], "category": r["name"], "type": kind, "amount": float(v)})
    return pd.DataFrame(rows)

if st.button("💾 Guardar Budget del año"):
    # Reemplaza budget del año en sesión
    budget_df2 = budget_df[budget_df["year"]!=year_sel].copy()
    budget_df2 = pd.concat([budget_df2, wide_budget_to_tidy(b_inc_edit, year_sel, "Ingreso"),
                                         wide_budget_to_tidy(b_exp_edit, year_sel, "Gasto")], ignore_index=True)
    st.session_state["budget"] = budget_df2
    budget_df = budget_df2
    st.success("Budget guardado en esta sesión.")

# Variances por mes
def sum_budget(df, year, kind):
    d = df[(df["year"]==year) & (df["type"]==kind)].groupby("month", as_index=False)["amount"].sum()
    d["month"] = pd.to_numeric(d["month"])
    return d.rename(columns={"amount": f"budget_{kind.lower()}"})
b_inc_tot = sum_budget(budget_df, year_sel, "Ingreso") if not budget_df.empty else pd.DataFrame({"month":[],"budget_ingreso":[]})
b_exp_tot = sum_budget(budget_df, year_sel, "Gasto") if not budget_df.empty else pd.DataFrame({"month":[],"budget_gasto":[]})

variance = pd.DataFrame({"month": range(1,13)}).merge(df_year[["month","income","expense","net"]], on="month", how="left").fillna(0)
if not b_inc_tot.empty: variance = variance.merge(b_inc_tot, on="month", how="left").fillna({"budget_ingreso":0})
else: variance["budget_ingreso"]=0
if not b_exp_tot.empty: variance = variance.merge(b_exp_tot, on="month", how="left").fillna({"budget_gasto":0})
else: variance["budget_gasto"]=0
variance["var_exp"] = variance["expense"] - variance["budget_gasto"]
variance["var_exp_pct"] = np.where(variance["budget_gasto"]>0, (variance["expense"]/variance["budget_gasto"] - 1), np.nan)

def color_for(pct):
    if pd.isna(pct): return "gray"
    if abs(pct) <= variance_green: return "green"
    if abs(pct) <= variance_yellow: return "orange"
    return "red"
variance["status"] = variance["var_exp_pct"].apply(color_for)

st.markdown("**Budget vs Actual — Gastos (mensual)**")
fig_var = px.bar(variance, x="month", y=["budget_gasto","expense"], barmode="group",
                 labels={"value":f"Monto ({show_currency})","month":"Mes","variable":""})
st.plotly_chart(fig_var, use_container_width=True)
st.dataframe(variance, use_container_width=True)

# ----------------------------------
# Exportación a Excel (backup de sesión)
# ----------------------------------
st.markdown("---")
st.subheader("⬇️ Exportar backup Excel (datos de la sesión)")
def build_excel_bytes():
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # Raw
        inc_by_cat2.to_excel(writer, sheet_name="INGRESOS_CAT", index=False)
        exp_by_cat2.to_excel(writer, sheet_name="GASTOS_CAT", index=False)
        st.session_state["transactions"].to_excel(writer, sheet_name="TRANSACCIONES", index=False)
        if not st.session_state["budget"].empty:
            st.session_state["budget"].to_excel(writer, sheet_name="BUDGET", index=False)
        # Aggregates
        totals_df2.to_excel(writer, sheet_name="TOTALS", index=False)
        variance.to_excel(writer, sheet_name="VARIANCE", index=False)
    out.seek(0)
    return out.read()

if st.button("Generar backup Excel"):
    data = build_excel_bytes()
    st.download_button("Descargar backup Excel", data, file_name="presupuesto_sesion.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Extra: modo presentación
st.markdown("---")
st.subheader("🕶️ Modo presentación")
hide_amounts = st.toggle("Ocultar montos en gráficos (para compartir pantalla)", False)
if hide_amounts:
    st.info("Ocultando etiquetas de valores. Los gráficos muestran tendencias sin números.")
