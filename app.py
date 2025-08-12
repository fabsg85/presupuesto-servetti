
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Presupuesto Familiar - Dashboard", layout="wide")

st.title("📊 Presupuesto Familiar Servetti — Dashboard interactivo")

st.write(
    "Subí tu archivo de Excel (formato actual) o usá el ejemplo para explorar KPIs, desglose por categorías, "
    "tendencias y un forecast de 6 meses. También podés agregar/editar/eliminar transacciones manuales."
)

def parse_year_sheet(df):
    df = df.copy()
    months = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
    month_row_idx = None
    for i in range(len(df)):
        row_vals = df.iloc[i].astype(str).tolist()
        if sum(m in row_vals for m in months) >= 6:
            month_row_idx = i
            break
    if month_row_idx is None:
        return None
    col_map = {}
    for m in months:
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
    idx_ing_total = df.index[df.iloc[:,label_col].astype(str).str.contains('TOTAL DE INGRESOS', na=False)].tolist()
    idx_gas_header = df.index[df.iloc[:,label_col].astype(str).str.match('GASTOS$', na=False)].tolist()
    idx_gas_total = df.index[df.iloc[:,label_col].astype(str).str.contains('TOTAL DE GASTOS', na=False)].tolist()

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
        totals['income'] = extract_total(idx_ing_total[0])
    if idx_gas_total:
        totals['expense'] = extract_total(idx_gas_total[0])
    result['totals'] = totals
    return result

def tidy_from_parsed(parsed):
    months_map = {'ENE':1,'FEB':2,'MAR':3,'ABR':4,'MAY':5,'JUN':6,'JUL':7,'AGO':8,'SEP':9,'OCT':10,'NOV':11,'DIC':12}
    rows = []
    cat_rows = []
    for year, content in parsed.items():
        for kind in ['income','expense']:
            s = content['totals'].get(kind, pd.Series())
            for m, val in s.items():
                if m in months_map:
                    rows.append({"year": int(year), "month": months_map[m], "kind": kind, "amount": val})
        # categories (expenses)
        exp_cat = content.get("expense_categories", pd.DataFrame())
        if not exp_cat.empty:
            for _, r in exp_cat.iterrows():
                for m, mon_num in months_map.items():
                    if m in exp_cat.columns:
                        val = r.get(m, np.nan)
                        if pd.notna(val):
                            cat_rows.append({
                                "year": int(year),
                                "month": mon_num,
                                "category": r["name"],
                                "amount": float(val)
                            })
    totals_df = pd.DataFrame(rows).dropna()
    by_cat = pd.DataFrame(cat_rows)
    return totals_df, by_cat

def load_workbook(file_bytes=None):
    if file_bytes is None:
        # Load sample included in repo if present
        return None, None, None
    xls = pd.ExcelFile(BytesIO(file_bytes))
    sheets = {}
    for name in xls.sheet_names:
        df = xls.parse(name, header=None)
        sheets[name] = parse_year_sheet(df)
    totals_df, exp_by_cat = tidy_from_parsed(sheets)
    return sheets, totals_df, exp_by_cat

# Sidebar
with st.sidebar:
    st.header("📥 Datos de entrada")
    uploaded = st.file_uploader("Subí el Excel (2023/2024/2025) con el formato actual", type=["xls","xlsx"])
    include_manual = st.toggle("Incluir transacciones manuales", True)
    st.markdown("---")
    st.subheader("💾 Persistencia de transacciones")
    st.caption("Podés descargar/subir tus transacciones para mantenerlas en el tiempo.")
    dl = st.session_state.get("transactions", pd.DataFrame())
    if not dl.empty:
        csv = dl.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar transacciones (CSV)", csv, file_name="transacciones.csv", mime="text/csv")
    up_tx = st.file_uploader("Subir transacciones (CSV)", type=["csv"], key="up_tx")
    if up_tx is not None:
        st.session_state["transactions"] = pd.read_csv(up_tx)

# Load data
sheets, totals_df, exp_by_cat = (None, None, None)
if uploaded is not None:
    sheets, totals_df, exp_by_cat = load_workbook(uploaded.read())

if totals_df is None or totals_df.empty:
    st.info("👋 Cargá tu Excel para ver KPIs y gráficos.")
else:
    # Build monthly net
    net = (totals_df.pivot_table(index=['year','month'], columns='kind', values='amount', aggfunc='sum')
           .reset_index())
    if 'income' not in net.columns: net['income'] = 0.0
    if 'expense' not in net.columns: net['expense'] = 0.0
    net['net'] = net['income'] - net['expense']

    # Transactions CRUD (manual adjustments)
    st.subheader("🧾 Transacciones manuales (opcional)")
    st.caption("Usá esta tabla para registrar movimientos detallados. Se agregan por encima del Excel (ajustes).")

    if "transactions" not in st.session_state:
        st.session_state["transactions"] = pd.DataFrame(columns=[
            "date","type","category","amount","account","notes"
        ])
    tx = st.session_state["transactions"]

    # Data editor
    edited = st.data_editor(
        tx,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("Fecha"),
            "type": st.column_config.SelectboxColumn("Tipo", options=["Ingreso","Gasto"]),
            "category": st.column_config.TextColumn("Categoría"),
            "amount": st.column_config.NumberColumn("Monto", step=1, format="%.2f"),
            "account": st.column_config.TextColumn("Cuenta / Medio"),
            "notes": st.column_config.TextColumn("Notas"),
        },
        key="tx_editor"
    )
    st.session_state["transactions"] = edited

    # Apply manual tx into aggregates (by month)
    if include_manual and not edited.empty:
        tmp = edited.dropna(subset=["date","amount"]).copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"])
        tmp["year"] = tmp["date"].dt.year
        tmp["month"] = tmp["date"].dt.month
        tmp["sign"] = np.where(tmp["type"]=="Ingreso", 1, -1)
        manual_agg = tmp.groupby(["year","month","type"], as_index=False)["amount"].sum()
        # Adjust net
        for _, r in manual_agg.iterrows():
            mask = (net["year"]==r["year"]) & (net["month"]==r["month"])
            if r["type"]=="Ingreso":
                net.loc[mask, "income"] = net.loc[mask, "income"].fillna(0) + r["amount"]
            else:
                net.loc[mask, "expense"] = net.loc[mask, "expense"].fillna(0) + r["amount"]
        net["net"] = net["income"] - net["expense"]

    # KPI zone (choose year)
    years = sorted(net["year"].unique())
    colA, colB = st.columns([1,3])
    with colA:
        year_sel = st.selectbox("Año", years, index=len(years)-1)
    df_year = net[net["year"]==year_sel].sort_values("month")

    ytd_income = df_year["income"].sum()
    ytd_expense = df_year["expense"].sum()
    ytd_net = df_year["net"].sum()
    avg_net = df_year["net"].mean()
    savings_rate = (1 - (ytd_expense / ytd_income)) * 100 if ytd_income else np.nan

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Ingresos YTD", f"{ytd_income:,.0f}")
    k2.metric("Gastos YTD", f"{ytd_expense:,.0f}")
    k3.metric("Ahorro Neto YTD", f"{ytd_net:,.0f}")
    k4.metric("Tasa de Ahorro", f"{savings_rate:,.1f}%")

    # Charts: income/expense/net
    line = px.line(df_year, x="month", y=["income","expense","net"], markers=True,
                   labels={"value":"Monto","month":"Mes","variable":"Serie"},
                   title=f"Evolución mensual — {year_sel}")
    st.plotly_chart(line, use_container_width=True)

    # Expense breakdown by category (if available)
    if exp_by_cat is not None and not exp_by_cat.empty:
        cats_year = exp_by_cat[exp_by_cat["year"]==year_sel]
        top_month = int(df_year.loc[df_year["expense"].idxmax(),"month"]) if not df_year.empty else 1
        col1,col2 = st.columns(2)
        with col1:
            bar = px.bar(cats_year.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(12),
                         x="category", y="amount", title=f"Top categorías de gasto — {year_sel}")
            st.plotly_chart(bar, use_container_width=True)
        with col2:
            cats_month = cats_year[cats_year["month"]==top_month]
            pie = px.pie(cats_month, names="category", values="amount", title=f"Distribución del mes más alto de gastos — Mes {top_month}")
            st.plotly_chart(pie, use_container_width=True)

    # Forecast (6 months) on net cash flow
    st.subheader("🔮 Forecast 6 meses")
    def make_forecast(series, periods=6):
        series = series.astype(float)
        # Simple Holt-Winters additive trend; guard min length
        if len(series.dropna()) < 6:
            return None
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit()
            fcast = fit.forecast(periods)
            return fcast
        except Exception:
            return None

    ser = df_year.set_index("month")["net"]
    fcast = make_forecast(ser, periods=6)
    if fcast is not None:
        fc_df = pd.DataFrame({"month": list(ser.index) + list(range(ser.index.max()+1, ser.index.max()+1+len(fcast))),
                              "net": list(ser.values) + list(fcast.values),
                              "type": ["hist"]*len(ser) + ["forecast"]*len(fcast)})
        area = px.area(fc_df, x="month", y="net", color="type", title="Cash flow neto — histórico y forecast (6 meses)")
        st.plotly_chart(area, use_container_width=True)

        st.caption(f"Proyección total 6m: {fcast.sum():,.0f} | Promedio mensual proyectado: {fcast.mean():,.0f}")
    else:
        st.info("Se necesitan al menos 6 puntos válidos para realizar el forecast.")

    # Balance acumulado (asumido sin saldo inicial)
    st.subheader("💰 Balance acumulado (sin saldo inicial)")
    df_year["balance_acum"] = df_year["net"].cumsum()
    area2 = px.area(df_year, x="month", y="balance_acum", title=f"Balance acumulado — {year_sel}")
    st.plotly_chart(area2, use_container_width=True)

    # Download KPIs dataset
    st.download_button(
        "⬇️ Descargar datos (Net mensual)",
        df_year.to_csv(index=False).encode("utf-8"),
        file_name=f"net_{year_sel}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("**Tip:** Para persistir transacciones en la nube, conectá una Google Sheet (gspread) o subí/descargá el CSV desde la barra lateral.")
