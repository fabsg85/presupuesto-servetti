
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Presupuesto Familiar — Supabase (sin Secrets)", layout="wide")
st.title("📊 Presupuesto Familiar — Supabase (sin Secrets)")
st.caption("Persistencia usando Supabase con **anon key** ingresada en la UI (sin Secrets).")

# ---------------------
# Supabase client (lazy)
# ---------------------
@st.cache_resource
def get_client(url, key):
    from supabase import create_client, Client
    return create_client(url, key)

def ensure_schema(sb):
    # Create tables if not exist (idempotent via rpc or execute SQL)
    # Use PostgREST: we'll attempt to select; if fails, show DDL to run in SQL editor.
    try:
        sb.table("settings").select("*").limit(1).execute()
    except Exception as e:
        st.error("No encuentro las tablas. Copiá y ejecutá este SQL en el **SQL Editor** de Supabase y reintenta:")
        st.code(\"\"\"
create table if not exists public.settings (key text primary key, value text);
create table if not exists public.income_cats (
  id bigserial primary key, year text, month int, category text, amount double precision,
  account text, currency text, tags text
);
create table if not exists public.expense_cats (
  id bigserial primary key, year text, month int, category text, amount double precision,
  account text, currency text, tags text
);
create table if not exists public.transactions (
  id bigserial primary key, date date, type text, category text, amount double precision,
  account text, currency text, tags text, notes text
);
create table if not exists public.budget (
  id bigserial primary key, year text, month int, category text, type text, amount double precision
);
create table if not exists public.goals (
  id bigserial primary key, name text, target_amount double precision, target_date date,
  monthly_contrib double precision, priority text, notes text
);
-- Policies abiertas tipo demo (ajusta en RLS)
alter table public.settings enable row level security;
alter table public.income_cats enable row level security;
alter table public.expense_cats enable row level security;
alter table public.transactions enable row level security;
alter table public.budget enable row level security;
alter table public.goals enable row level security;
create policy "anon_all" on public.settings for all using (true) with check (true);
create policy "anon_all" on public.income_cats for all using (true) with check (true);
create policy "anon_all" on public.expense_cats for all using (true) with check (true);
create policy "anon_all" on public.transactions for all using (true) with check (true);
create policy "anon_all" on public.budget for all using (true) with check (true);
create policy "anon_all" on public.goals for all using (true) with check (true);
        \"\"\")
        st.stop()

with st.sidebar:
    st.header("🔌 Conexión Supabase (sin Secrets)")
    url = st.text_input("Supabase URL", placeholder="https://xxxxx.supabase.co")
    key = st.text_input("Anon Key", type="password", placeholder="eyJhbGciOi...")
    st.caption("Crea un proyecto gratis en supabase.com → Settings → Project API para copiar **URL** y **anon key**.")

if not url or not key:
    st.info("Ingresá URL y anon key de Supabase para continuar.")
    st.stop()

sb = get_client(url, key)
ensure_schema(sb)

# ---------------------
# Settings
# ---------------------
def get_settings():
    res = sb.table("settings").select("*").execute().data
    df = pd.DataFrame(res)
    if df.empty:
        df = pd.DataFrame({"key":["base_currency","fx_usd_uyu","variance_green","variance_yellow"],
                           "value":["UYU","40","0.05","0.15"]})
        for _, r in df.iterrows():
            sb.table("settings").insert({"key":r["key"],"value":r["value"]}).execute()
    return pd.DataFrame(sb.table("settings").select("*").execute().data)

def set_settings(df):
    sb.table("settings").delete().neq("key","").execute()
    payload = df.to_dict(orient="records")
    if payload:
        sb.table("settings").insert(payload).execute()

settings_df = get_settings()
smap = settings_df.set_index("key")["value"].to_dict()
base_currency = smap.get("base_currency","UYU")
fx = float(smap.get("fx_usd_uyu","40"))
variance_green = float(smap.get("variance_green","0.05"))
variance_yellow = float(smap.get("variance_yellow","0.15"))

with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Configuración")
    base_currency = st.selectbox("Moneda base", ["UYU","USD"], index=0 if base_currency=="UYU" else 1)
    fx = st.number_input("Tipo de cambio (USD→UYU)", min_value=1.0, value=float(fx), step=0.5)
    variance_green = st.number_input("Umbral verde (desvío ≤)", min_value=0.0, value=float(variance_green), step=0.01)
    variance_yellow = st.number_input("Umbral amarillo (desvío ≤)", min_value=0.0, value=float(variance_yellow), step=0.01)
    if st.button("💾 Guardar configuración"):
        cfg = pd.DataFrame({"key":["base_currency","fx_usd_uyu","variance_green","variance_yellow"],
                            "value":[base_currency, str(fx), str(variance_green), str(variance_yellow)]})
        set_settings(cfg)
        st.success("Configuración guardada.")

# ---------------------
# CRUD helpers
# ---------------------
def read_table(name):
    return pd.DataFrame(sb.table(name).select("*").execute().data)

def write_table(name, df):
    sb.table(name).delete().neq("id", -1).execute()
    records = df.to_dict(orient="records")
    if records:
        sb.table(name).insert(records).execute()

# Load data
income_df  = read_table("income_cats")
expense_df = read_table("expense_cats")
tx_df      = read_table("transactions")
budget_df  = read_table("budget")
goals_df   = read_table("goals")

MONTHS = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
REV_MONTHS = {i+1:m for i,m in enumerate(MONTHS)}
MONTHS_MAP = {m:i+1 for i,m in enumerate(MONTHS)}

def to_wide(df, year):
    d = df[df.get("year","").astype(str)==str(year)].copy()
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

def wide_to_tidy(wide, year):
    rows = []
    for _, r in wide.iterrows():
        for m in MONTHS:
            v = r.get(m, 0.0)
            if pd.notna(v) and float(v) != 0.0:
                rows.append({"year": str(year), "month": MONTHS_MAP[m], "category": r["name"], "amount": float(v),
                             "account":"SB", "currency":"UYU", "tags":""})
    return pd.DataFrame(rows)

def convert_amount(amount, cur_from):
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
        inc["amount_base"] = [convert_amount(a, c) for a,c in zip(inc["amount"], inc.get("currency","UYU"))]
    if not exp.empty:
        exp["amount_base"] = [convert_amount(a, c) for a,c in zip(exp["amount"], exp.get("currency","UYU"))]
    inc_tot = inc.groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"income"})
    exp_tot = exp.groupby(["year","month"], as_index=False)["amount_base"].sum().rename(columns={"amount_base":"expense"})
    merged = pd.merge(inc_tot, exp_tot, on=["year","month"], how="outer").fillna(0)
    if not tx_df.empty:
        t = tx_df.dropna(subset=["date","amount"]).copy()
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t = t.dropna(subset=["date"])
        t["year"] = t["date"].dt.year.astype(str)
        t["month"] = t["date"].dt.month
        t["amount_base"] = [convert_amount(a, c) for a,c in zip(t["amount"], t.get("currency","UYU"))]
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

# Editors
st.subheader("🧾 Editores por año")
years_present = sorted(list(set(income_df.get("year",[])) | set(expense_df.get("year",[])) | set(pd.to_datetime(tx_df.get("date",[]), errors='coerce').dropna().dt.year.astype(str))))
if not years_present: years_present = [str(datetime.now().year)]
year_sel = st.selectbox("Año", years_present, index=len(years_present)-1)

inc_w = to_wide(income_df, year_sel)
exp_w = to_wide(expense_df, year_sel)

c1,c2 = st.columns(2)
with c1:
    st.markdown("**Ingresos (categoría x mes)**")
    inc_edit = st.data_editor(inc_w, num_rows="dynamic", use_container_width=True)
with c2:
    st.markdown("**Gastos (categoría x mes)**")
    exp_edit = st.data_editor(exp_w, num_rows="dynamic", use_container_width=True)

if st.button("💾 Guardar ingresos/gastos del año"):
    inc_new = wide_to_tidy(inc_edit, year_sel)
    exp_new = wide_to_tidy(exp_edit, year_sel)
    # Replace year in DB
    sb.table("income_cats").delete().eq("year", str(year_sel)).execute()
    sb.table("expense_cats").delete().eq("year", str(year_sel)).execute()
    if not inc_new.empty:
        sb.table("income_cats").insert(inc_new.to_dict(orient="records")).execute()
    if not exp_new.empty:
        sb.table("expense_cats").insert(exp_new.to_dict(orient="records")).execute()
    st.success("Guardado.")

# Transactions
st.subheader("🧾 Transacciones manuales")
tx_view = tx_df.copy()
tx_edit = st.data_editor(tx_view, num_rows="dynamic", use_container_width=True)
if st.button("💾 Guardar transacciones"):
    # Simple approach: replace all
    sb.table("transactions").delete().neq("id",-1).execute()
    if not tx_edit.empty:
        sb.table("transactions").insert(tx_edit.to_dict(orient="records")).execute()
    st.success("Transacciones guardadas.")

# KPIs / Charts
st.markdown("---")
st.subheader("📌 KPIs y gráficos")
totals = build_totals(income_df, expense_df, tx_df)
year_opts = sorted(totals["year"].unique())
year_kpi = st.selectbox("Año KPIs", year_opts, index=len(year_opts)-1)
dfy = totals[totals["year"]==year_kpi].sort_values("month")
dfy["net_ma3"] = dfy["net"].rolling(3).mean()

c1,c2,c3,c4 = st.columns(4)
c1.metric(f"Ingresos YTD ({base_currency})", f"{dfy['income'].sum():,.0f}")
c2.metric(f"Gastos YTD ({base_currency})", f"{dfy['expense'].sum():,.0f}")
c3.metric(f"Neto YTD ({base_currency})", f"{dfy['net'].sum():,.0f}")
srate = (1 - (dfy['expense'].sum()/dfy['income'].sum()))*100 if dfy['income'].sum()>0 else np.nan
c4.metric("Tasa de Ahorro", f"{srate:,.1f}%")

line = px.line(dfy, x="month", y=["income","expense","net","net_ma3"], markers=True)
st.plotly_chart(line, use_container_width=True)

st.success("Persistencia activa en Supabase (sin Secrets).")

