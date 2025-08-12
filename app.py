
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Presupuesto Familiar — Plus", layout="wide")

st.title("📊 Presupuesto Familiar — Dashboard avanzado")
st.caption("Versión con editor tipo Excel, KPIs ampliados, gráficas mejoradas, forecast por serie y alertas.")

# -----------------------------
# Parsing helpers
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
                                "year": int(year), "month": mon_num, "category": r["name"], "amount": float(val)
                            })
        inc_cat = content.get("income_categories", pd.DataFrame())
        if not inc_cat.empty:
            for _, r in inc_cat.iterrows():
                for m, mon_num in MONTHS_MAP.items():
                    if m in inc_cat.columns:
                        val = r.get(m, np.nan)
                        if pd.notna(val):
                            cat_inc_rows.append({
                                "year": int(year), "month": mon_num, "category": r["name"], "amount": float(val)
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
    # Requires >= 6 data points. If multi-year (>=12), enable seasonality=12
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

# -----------------------------
# UI: Sidebar
# -----------------------------
with st.sidebar:
    st.header("📥 Datos de entrada")
    uploaded = st.file_uploader("Subí el Excel con el formato actual (2023/2024/2025)", type=["xls","xlsx"])

    st.markdown("---")
    st.subheader("🎯 Objetivos & Alertas")
    target_savings = st.number_input("Objetivo de ahorro mensual (aprox.)", min_value=0.0, value=0.0, step=100.0)
    alert_threshold = st.slider("Alerta cuando el gasto mensual exceda su media 3m por (%)", 5, 100, 25)
    compare_prev_year = st.toggle("Comparar contra mismo mes del año anterior", True)

    st.markdown("---")
    st.subheader("💾 Exportar")
    st.caption("Podés descargar CSVs con los datos procesados o el editor por año.")

# -----------------------------
# Load & session state
# -----------------------------
if "edited_tables" not in st.session_state:
    st.session_state["edited_tables"] = {}  # per year: {"income": df, "expense": df}

sheets, totals_df, exp_by_cat, inc_by_cat = (None, None, None, None)
if uploaded is not None:
    sheets, totals_df, exp_by_cat, inc_by_cat = load_workbook(uploaded.read())

if totals_df is None or totals_df.empty:
    st.info("👋 Cargá tu Excel para ver KPIs y usar el editor tipo Excel.")
    st.stop()

# -----------------------------
# Editor tipo Excel (por año)
# -----------------------------
years = sorted({int(y) for y in totals_df["year"].unique()})
st.subheader("🧾 Editor tipo Excel — por año")
year_tab = st.selectbox("Año a editar", years, index=len(years)-1)

# Get initial tables for the selected year
def get_year_tables(year):
    # Build wide income/expense tables from category details if available; if not, from totals only
    inc = inc_by_cat[inc_by_cat["year"]==year] if inc_by_cat is not None else pd.DataFrame()
    exp = exp_by_cat[exp_by_cat["year"]==year] if exp_by_cat is not None else pd.DataFrame()

    def to_wide(df, label):
        if df is None or df.empty:
            return pd.DataFrame(columns=["name"] + MONTHS)
        wide = df.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
        wide.columns = [REV_MONTHS_MAP.get(c, c) for c in wide.columns]
        wide = wide.reset_index().rename(columns={"category":"name"})
        # Ensure all months present
        for m in MONTHS:
            if m not in wide.columns:
                wide[m] = 0.0
        return wide[["name"] + MONTHS]

    inc_w = to_wide(inc, "Ingreso")
    exp_w = to_wide(exp, "Gasto")
    return inc_w, exp_w

if year_tab not in st.session_state["edited_tables"]:
    inc_w, exp_w = get_year_tables(year_tab)
    st.session_state["edited_tables"][year_tab] = {"income": inc_w, "expense": exp_w}

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Ingresos (editor)**")
    inc_edit = st.data_editor(
        st.session_state["edited_tables"][year_tab]["income"],
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
        st.session_state["edited_tables"][year_tab]["expense"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={m: st.column_config.NumberColumn(m, format="%.2f", step=1) for m in MONTHS} | {
            "name": st.column_config.TextColumn("Categoría")
        },
        key=f"exp_editor_{year_tab}"
    )

# Save back to session
st.session_state["edited_tables"][year_tab]["income"]  = inc_edit
st.session_state["edited_tables"][year_tab]["expense"] = exp_edit

# Rebuild aggregates from editors (override parsed for this year)
def rebuild_totals_from_editors(totals_df, inc_by_cat, exp_by_cat):
    # Copy
    tdf = totals_df.copy()
    ibc = inc_by_cat.copy() if inc_by_cat is not None else pd.DataFrame(columns=["year","month","category","amount"])
    ebc = exp_by_cat.copy() if exp_by_cat is not None else pd.DataFrame(columns=["year","month","category","amount"])

    for y, data in st.session_state["edited_tables"].items():
        # Turn wide into tidy
        for kind in ["income","expense"]:
            wide = data[kind]
            tidy_rows = []
            for _, r in wide.iterrows():
                for m in MONTHS:
                    val = r.get(m, 0.0)
                    if pd.notna(val) and float(val) != 0.0:
                        tidy_rows.append({"year": int(y), "month": MONTHS_MAP[m], "category": r["name"], "amount": float(val)})
            tidy = pd.DataFrame(tidy_rows)
            if kind == "income":
                ibc = ibc[ibc["year"]!=int(y)]
                ibc = pd.concat([ibc, tidy], ignore_index=True)
            else:
                ebc = ebc[ebc["year"]!=int(y)]
                ebc = pd.concat([ebc, tidy], ignore_index=True)

    # Rebuild totals from categories
    inc_tot = ibc.groupby(["year","month"], as_index=False)["amount"].sum().rename(columns={"amount":"income"})
    exp_tot = ebc.groupby(["year","month"], as_index=False)["amount"].sum().rename(columns={"amount":"expense"})
    merged = pd.merge(inc_tot, exp_tot, on=["year","month"], how="outer").fillna(0)
    merged = merged.melt(id_vars=["year","month"], var_name="kind", value_name="amount")
    # Remove original rows for edited years and add new
    edited_years = list(st.session_state["edited_tables"].keys())
    tdf = tdf[~tdf["year"].isin(edited_years)]
    tdf = pd.concat([tdf, merged], ignore_index=True)
    return tdf, ebc, ibc

totals_df2, exp_by_cat2, inc_by_cat2 = rebuild_totals_from_editors(totals_df, inc_by_cat, exp_by_cat)

# -----------------------------
# KPIs ampliados
# -----------------------------
net = (totals_df2.pivot_table(index=["year","month"], columns="kind", values="amount", aggfunc="sum")
       .reset_index())
for col in ["income","expense"]:
    if col not in net.columns: net[col] = 0.0
net["net"] = net["income"] - net["expense"]

year_sel = st.selectbox("Año para KPIs y gráficos", years, index=len(years)-1, key="kpi_year")
df_year = net[net["year"]==year_sel].sort_values("month")

# Moving averages
df_year["net_ma3"] = df_year["net"].rolling(3).mean()
df_year["exp_ma3"] = df_year["expense"].rolling(3).mean()
df_year["inc_ma3"] = df_year["income"].rolling(3).mean()

ytd_income  = df_year["income"].sum()
ytd_expense = df_year["expense"].sum()
ytd_net     = df_year["net"].sum()
avg_net     = df_year["net"].mean()
savings_rate = (1 - (ytd_expense / ytd_income)) * 100 if ytd_income else np.nan
best_month = int(df_year.loc[df_year["net"].idxmax(),"month"]) if not df_year.empty else None
worst_month = int(df_year.loc[df_year["net"].idxmin(),"month"]) if not df_year.empty else None
volatility = df_year["net"].std()

k1,k2,k3,k4 = st.columns(4)
k1.metric("Ingresos YTD", f"{ytd_income:,.0f}")
k2.metric("Gastos YTD", f"{ytd_expense:,.0f}")
k3.metric("Ahorro Neto YTD", f"{ytd_net:,.0f}")
k4.metric("Tasa de Ahorro", f"{savings_rate:,.1f}%")

k5,k6,k7,k8 = st.columns(4)
k5.metric("Prom. neto mensual", f"{avg_net:,.0f}")
k6.metric("Volatilidad (σ neto)", f"{volatility:,.0f}")
k7.metric("Mejor mes (neto)", f"{best_month if best_month else '-'}")
k8.metric("Peor mes (neto)", f"{worst_month if worst_month else '-'}")

# -----------------------------
# Gráficas mejoradas
# -----------------------------
st.markdown("### 📈 Evolución mensual y medias móviles")
fig_line = px.line(df_year, x="month", y=["income","expense","net","net_ma3"], markers=True,
                   labels={"value":"Monto","month":"Mes","variable":"Serie"})
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("### 📊 Ingresos vs Gastos (stacked)")
bar = px.bar(df_year.melt(id_vars=["month"], value_vars=["income","expense"], var_name="tipo", value_name="monto"),
             x="month", y="monto", color="tipo", barmode="group")
st.plotly_chart(bar, use_container_width=True)

# Heatmap de gasto por categoría
if not exp_by_cat2.empty:
    cats_year = exp_by_cat2[exp_by_cat2["year"]==year_sel]
    heat = cats_year.pivot_table(index="category", columns="month", values="amount", aggfunc="sum").fillna(0)
    heat = heat[sorted(heat.columns)]
    im = px.imshow(heat, aspect="auto", labels=dict(x="Mes", y="Categoría", color="Gasto"))
    st.markdown("### 🔥 Heatmap de gastos por categoría")
    st.plotly_chart(im, use_container_width=True)

# Treemap top categorías
if not exp_by_cat2.empty:
    top_cats = (exp_by_cat2[exp_by_cat2["year"]==year_sel]
                .groupby("category", as_index=False)["amount"].sum()
                .sort_values("amount", ascending=False).head(20))
    tree = px.treemap(top_cats, path=["category"], values="amount", title="Top 20 categorías del año")
    st.plotly_chart(tree, use_container_width=True)

# Waterfall del neto
wf = go.Figure(go.Waterfall(
    x=[REV_MONTHS_MAP[m] for m in df_year["month"]],
    measure=["relative"]*len(df_year),
    y=df_year["net"],
    connector={"line":{"dash":"dot"}},
))
wf.update_layout(title="💧 Waterfall — Cash flow neto por mes", showlegend=False)
st.plotly_chart(wf, use_container_width=True)

# Comparación con año anterior (opcional)
if compare_prev_year and (year_sel-1 in years):
    prev = net[net["year"]==year_sel-1].sort_values("month")
    comp = df_year[["month","income","expense","net"]].merge(prev[["month","income","expense","net"]], on="month", suffixes=("", "_prev"), how="left")
    comp_fig = px.line(comp, x="month", y=["net","net_prev"], markers=True, title=f"Comparativa neto {year_sel} vs {year_sel-1}")
    st.plotly_chart(comp_fig, use_container_width=True)

# -----------------------------
# Forecast por serie e insight
# -----------------------------
st.markdown("### 🔮 Forecast (6 meses) — ingresos, gastos y neto")
multi_year = len(net["year"].unique()) >= 2
ser_inc = df_year.set_index("month")["income"]
ser_exp = df_year.set_index("month")["expense"]
ser_net = df_year.set_index("month")["net"]

f_inc = forecast_series(ser_inc, 6, seasonal=multi_year)
f_exp = forecast_series(ser_exp, 6, seasonal=multi_year)
f_net = forecast_series(ser_net, 6, seasonal=multi_year)

def plot_forecast(hist, fcast, title):
    if fcast is None:
        st.info(f"No hay suficientes datos para {title}.")
        return
    fc_df = pd.DataFrame({
        "month": list(hist.index) + list(range(int(hist.index.max())+1, int(hist.index.max())+1+len(fcast))),
        "value": list(hist.values) + list(fcast.values),
        "tipo": ["hist"]*len(hist) + ["forecast"]*len(fcast)
    })
    fig = px.area(fc_df, x="month", y="value", color="tipo", title=title)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Proyección 6m total: {fcast.sum():,.0f} | Promedio mensual proyectado: {fcast.mean():,.0f}")

cols = st.columns(3)
with cols[0]: plot_forecast(ser_inc, f_inc, "Ingresos")
with cols[1]: plot_forecast(ser_exp, f_exp, "Gastos")
with cols[2]: plot_forecast(ser_net, f_net, "Neto")

# -----------------------------
# Alertas y sugerencias
# -----------------------------
st.markdown("### 🚨 Alertas inteligentes")
alerts = []

# 1) Gasto mensual vs media móvil 3m por categoría
if not exp_by_cat2.empty:
    cy = exp_by_cat2[exp_by_cat2["year"]==year_sel]
    for m in sorted(cy["month"].unique()):
        cm = cy[cy["month"]==m]
        # MA3 por categoría
        for cat, grp in cy.groupby("category"):
            series = grp.set_index("month").sort_index()["amount"]
            ma3 = series.rolling(3).mean().get(m, np.nan)
            val = series.get(m, np.nan)
            if pd.notna(val) and pd.notna(ma3) and ma3 > 0:
                pct = (val/ma3 - 1)*100
                if pct >= alert_threshold:
                    alerts.append(f"Mes {REV_MONTHS_MAP[m]}: **{cat}** está {pct:.0f}% sobre su media móvil 3m ({val:,.0f} vs {ma3:,.0f}).")

# 2) Neto vs objetivo de ahorro
if target_savings and target_savings > 0 and not df_year.empty:
    for _, r in df_year.iterrows():
        diff = r["net"] - target_savings
        if diff < 0:
            alerts.append(f"Mes {REV_MONTHS_MAP[int(r['month'])]}: neto {r['net']:,.0f} por debajo del objetivo ({target_savings:,.0f}) en {abs(diff):,.0f}.")

# 3) Gasto mes actual vs mismo mes año previo
if compare_prev_year and (year_sel-1 in years):
    prev = net[net["year"]==year_sel-1]
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
        st.info(f"…y {len(alerts)-8} alertas más. Ajustá el umbral en la barra lateral.")
else:
    st.success("Sin alertas con los parámetros actuales.")

# -----------------------------
# Descargas
# -----------------------------
st.markdown("---")
st.subheader("⬇️ Descargas")
# CSVs procesados
st.download_button("Totales (income/expense) — tidy CSV", totals_df2.to_csv(index=False).encode("utf-8"), file_name="totales_tidy.csv", mime="text/csv")
if not exp_by_cat2.empty:
    st.download_button("Gastos por categoría — tidy CSV", exp_by_cat2.to_csv(index=False).encode("utf-8"), file_name="gastos_categoria_tidy.csv", mime="text/csv")
if not inc_by_cat2.empty:
    st.download_button("Ingresos por categoría — tidy CSV", inc_by_cat2.to_csv(index=False).encode("utf-8"), file_name="ingresos_categoria_tidy.csv", mime="text/csv")

# Exportar editores del año seleccionado
wide_pack = {
    "income": st.session_state["edited_tables"][year_tab]["income"].to_csv(index=False),
    "expense": st.session_state["edited_tables"][year_tab]["expense"].to_csv(index=False),
}
st.download_button(f"Editor {year_sel} — Ingresos (CSV)", wide_pack["income"].encode("utf-8"), file_name=f"ingresos_{year_sel}.csv", mime="text/csv")
st.download_button(f"Editor {year_sel} — Gastos (CSV)", wide_pack["expense"].encode("utf-8"), file_name=f"gastos_{year_sel}.csv", mime="text/csv")

st.caption("Los cambios del editor viven en sesión. Para persistirlos, descargá y luego unificalos en tu Excel, o mantené un CSV maestro. Si querés, puedo agregar exportación a Excel por año.")
