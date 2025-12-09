import hashlib
import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = "finance.db"

st.set_page_config(
    page_title="Finance CRM Dashboard",
    page_icon="💰",
    layout="wide",
)

# ----------------------------
# Database helpers
# ----------------------------


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS User (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            passwordHash TEXT NOT NULL,
            createdAt TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            year INTEGER NOT NULL,
            currency TEXT NOT NULL,
            savingsTargetPercentage REAL NOT NULL,
            FOREIGN KEY(userId) REFERENCES User(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Category (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('INGRESO', 'GASTO')),
            name TEXT NOT NULL,
            fixedVariable TEXT CHECK(fixedVariable IN ('FIJO', 'VARIABLE')),
            essential INTEGER DEFAULT 0,
            monthlyBudget REAL,
            UNIQUE(userId, type, name),
            FOREIGN KEY(userId) REFERENCES User(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Transaction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            date TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('INGRESO', 'GASTO')),
            categoryId INTEGER NOT NULL,
            amount REAL NOT NULL,
            paymentMethod TEXT NOT NULL,
            notes TEXT,
            createdAt TEXT NOT NULL,
            FOREIGN KEY(userId) REFERENCES User(id),
            FOREIGN KEY(categoryId) REFERENCES Category(id)
        );
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def seed_data():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM User")
    if cur.fetchone()[0] > 0:
        conn.close()
        return

    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO User (name, email, passwordHash, createdAt) VALUES (?, ?, ?, ?)",
        ("Demo User", "demo@user.com", hash_password("demo1234"), now),
    )
    user_id = cur.lastrowid
    cur.execute(
        "INSERT INTO Config (userId, year, currency, savingsTargetPercentage) VALUES (?, ?, ?, ?)",
        (user_id, 2026, "UYU", 0.2),
    )

    categories = [
        ("INGRESO", "Sueldo Fabri", None, 1, None),
        ("INGRESO", "Sueldo Dani", None, 1, None),
        ("INGRESO", "Otros ingresos", None, 0, None),
        ("GASTO", "Alquiler", "FIJO", 1, 25000),
        ("GASTO", "Electricidad", "FIJO", 1, 4500),
        ("GASTO", "Agua", "FIJO", 1, 2200),
        ("GASTO", "Internet", "FIJO", 1, 2000),
        ("GASTO", "Supermercado", "VARIABLE", 1, 15000),
        ("GASTO", "Salidas / Comida afuera", "VARIABLE", 0, 8000),
        ("GASTO", "Transporte", "VARIABLE", 1, 6000),
        ("GASTO", "Mascotas", "VARIABLE", 0, 4000),
        ("GASTO", "Gastos NO reintegrables", "VARIABLE", 0, 5000),
    ]
    for cat in categories:
        cur.execute(
            "INSERT INTO Category (userId, type, name, fixedVariable, essential, monthlyBudget) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, *cat),
        )

    # sample transactions for Oct-Dec 2026
    transactions = [
        ("2026-10-01", "INGRESO", "Sueldo Fabri", 65000, "Santander", "Sueldo mensual"),
        ("2026-10-05", "INGRESO", "Sueldo Dani", 52000, "Santander", None),
        ("2026-10-06", "GASTO", "Alquiler", 25000, "Débito", None),
        ("2026-10-07", "GASTO", "Supermercado", 12000, "Crédito", None),
        ("2026-10-10", "GASTO", "Salidas / Comida afuera", 4500, "Crédito", "Cena"),
        ("2026-10-12", "GASTO", "Transporte", 3000, "Santander", None),
        ("2026-11-01", "INGRESO", "Sueldo Fabri", 65000, "Santander", None),
        ("2026-11-05", "INGRESO", "Sueldo Dani", 52000, "Santander", None),
        ("2026-11-06", "GASTO", "Alquiler", 25000, "Débito", None),
        ("2026-11-07", "GASTO", "Supermercado", 14500, "Crédito", None),
        ("2026-11-11", "GASTO", "Electricidad", 4100, "Débito", None),
        ("2026-11-15", "GASTO", "Salidas / Comida afuera", 6000, "Crédito", None),
        ("2026-11-20", "GASTO", "Transporte", 3200, "Santander", None),
        ("2026-12-01", "INGRESO", "Sueldo Fabri", 65000, "Santander", None),
        ("2026-12-05", "INGRESO", "Sueldo Dani", 52000, "Santander", None),
        ("2026-12-06", "GASTO", "Alquiler", 25000, "Débito", None),
        ("2026-12-07", "GASTO", "Supermercado", 16000, "Crédito", None),
        ("2026-12-09", "GASTO", "Internet", 2000, "Crédito", None),
        ("2026-12-10", "GASTO", "Electricidad", 4300, "Crédito", None),
        ("2026-12-12", "GASTO", "Salidas / Comida afuera", 7000, "Crédito", None),
        ("2026-12-15", "GASTO", "Gastos NO reintegrables", 3500, "Crédito", "Consultas"),
    ]

    cat_lookup = {
        name: cur.execute(
            "SELECT id FROM Category WHERE userId=? AND name=?",
            (user_id, name),
        ).fetchone()[0]
        for name in [c[1] for c in categories]
    }

    for tx in transactions:
        cat_id = cat_lookup[tx[2]]
        cur.execute(
            """
            INSERT INTO Transaction (userId, date, type, categoryId, amount, paymentMethod, notes, createdAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                tx[0],
                tx[1],
                cat_id,
                tx[3],
                tx[4],
                tx[5],
                now,
            ),
        )

    conn.commit()
    conn.close()


init_db()
seed_data()

# ----------------------------
# Helpers
# ----------------------------


def get_user_by_email(email: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, passwordHash FROM User WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    return row


def get_config(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, year, currency, savingsTargetPercentage FROM Config WHERE userId=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def list_categories(user_id: int, type_filter: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    if type_filter:
        cur.execute(
            "SELECT id, type, name, fixedVariable, essential, monthlyBudget FROM Category WHERE userId=? AND type=? ORDER BY name",
            (user_id, type_filter),
        )
    else:
        cur.execute(
            "SELECT id, type, name, fixedVariable, essential, monthlyBudget FROM Category WHERE userId=? ORDER BY type, name",
            (user_id,),
        )
    rows = cur.fetchall()
    conn.close()
    return rows


def list_transactions(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.id, t.date, t.type, c.name, t.amount, t.paymentMethod, t.notes, c.fixedVariable, c.essential
        FROM Transaction t
        JOIN Category c ON t.categoryId = c.id
        WHERE t.userId=?
        ORDER BY date DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def upsert_config(user_id: int, year: int, currency: str, savings_target: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM Config WHERE userId=?", (user_id,))
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE Config SET year=?, currency=?, savingsTargetPercentage=? WHERE id=?",
            (year, currency, savings_target, existing[0]),
        )
    else:
        cur.execute(
            "INSERT INTO Config (userId, year, currency, savingsTargetPercentage) VALUES (?, ?, ?, ?)",
            (user_id, year, currency, savings_target),
        )
    conn.commit()
    conn.close()


def create_or_update_category(user_id: int, category_id: Optional[int], type_: str, name: str, fixed_var: Optional[str], essential: bool, budget: Optional[float]):
    conn = get_conn()
    cur = conn.cursor()
    if category_id:
        cur.execute(
            "UPDATE Category SET type=?, name=?, fixedVariable=?, essential=?, monthlyBudget=? WHERE id=? AND userId=?",
            (type_, name, fixed_var, int(essential), budget, category_id, user_id),
        )
    else:
        cur.execute(
            "INSERT INTO Category (userId, type, name, fixedVariable, essential, monthlyBudget) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, type_, name, fixed_var, int(essential), budget),
        )
    conn.commit()
    conn.close()


def delete_category(user_id: int, category_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM Transaction WHERE categoryId=?", (category_id,))
    if cur.fetchone()[0] > 0:
        conn.close()
        return False
    cur.execute("DELETE FROM Category WHERE id=? AND userId=?", (category_id, user_id))
    conn.commit()
    conn.close()
    return True


def create_or_update_transaction(
    user_id: int,
    tx_id: Optional[int],
    date_value: date,
    type_: str,
    category_id: int,
    amount: float,
    payment_method: str,
    notes: Optional[str],
):
    conn = get_conn()
    cur = conn.cursor()
    if tx_id:
        cur.execute(
            "UPDATE Transaction SET date=?, type=?, categoryId=?, amount=?, paymentMethod=?, notes=? WHERE id=? AND userId=?",
            (
                date_value.isoformat(),
                type_,
                category_id,
                amount,
                payment_method,
                notes,
                tx_id,
                user_id,
            ),
        )
    else:
        cur.execute(
            """
            INSERT INTO Transaction (userId, date, type, categoryId, amount, paymentMethod, notes, createdAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                date_value.isoformat(),
                type_,
                category_id,
                amount,
                payment_method,
                notes,
                datetime.utcnow().isoformat(),
            ),
        )
    conn.commit()
    conn.close()


def delete_transaction(user_id: int, tx_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM Transaction WHERE id=? AND userId=?", (tx_id, user_id))
    conn.commit()
    conn.close()


def export_transactions_csv(user_id: int) -> pd.DataFrame:
    rows = list_transactions(user_id)
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "date",
            "type",
            "category",
            "amount",
            "paymentMethod",
            "notes",
            "fixedVariable",
            "essential",
        ],
    )
    return df


# ----------------------------
# Auth
# ----------------------------


def login_form():
    st.header("Iniciar sesión")
    with st.form("login_form"):
        email = st.text_input("Email", value="demo@user.com")
        password = st.text_input("Contraseña", type="password", value="demo1234")
        submitted = st.form_submit_button("Ingresar")
    if submitted:
        user = get_user_by_email(email)
        if user and user[3] == hash_password(password):
            st.session_state.user = {"id": user[0], "name": user[1], "email": user[2]}
            st.success("Bienvenido de nuevo")
        else:
            st.error("Credenciales inválidas")


# ----------------------------
# Dashboard helpers
# ----------------------------


def load_transactions_df(user_id: int) -> pd.DataFrame:
    rows = list_transactions(user_id)
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "date",
            "type",
            "category",
            "amount",
            "paymentMethod",
            "notes",
            "fixedVariable",
            "essential",
        ],
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def compute_monthly_totals(df: pd.DataFrame, year: int):
    monthly = (
        df[df["year"] == year]
        .groupby(["month", "type"])
        .agg({"amount": "sum"})
        .reset_index()
        .pivot(index="month", columns="type", values="amount")
        .fillna(0)
        .reset_index()
    )
    monthly.columns.name = None
    return monthly


def get_budget_totals(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT SUM(monthlyBudget) FROM Category WHERE userId=? AND type='GASTO' AND monthlyBudget IS NOT NULL",
        (user_id,),
    )
    total = cur.fetchone()[0] or 0
    conn.close()
    return total


def get_category_totals(df: pd.DataFrame, month: int, year: int):
    filtered = df[(df["month"] == month) & (df["year"] == year) & (df["type"] == "GASTO")]
    if filtered.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return filtered.groupby("category").agg({"amount": "sum"}).reset_index()


def savings_percentage(income: float, expense: float) -> float:
    if income == 0:
        return 0
    return (income - expense) / income


# ----------------------------
# UI components
# ----------------------------


def dashboard_page(user):
    config = get_config(user["id"])
    year = config[1] if config else datetime.utcnow().year
    df = load_transactions_df(user["id"])
    month_names = {
        1: "Enero",
        2: "Febrero",
        3: "Marzo",
        4: "Abril",
        5: "Mayo",
        6: "Junio",
        7: "Julio",
        8: "Agosto",
        9: "Setiembre",
        10: "Octubre",
        11: "Noviembre",
        12: "Diciembre",
    }

    if df.empty:
        st.info("No hay transacciones aún")
        return

    available_years = sorted(df["year"].unique())
    selected_year = st.selectbox("Año", available_years, index=available_years.index(year) if year in available_years else 0)
    available_months = sorted(df[df["year"] == selected_year]["month"].unique())
    selected_month = st.selectbox("Mes", available_months, format_func=lambda m: month_names.get(m, str(m)))

    monthly_budget_total = get_budget_totals(user["id"])
    month_df = df[(df["month"] == selected_month) & (df["year"] == selected_year)]
    total_income = month_df[month_df["type"] == "INGRESO"]["amount"].sum()
    total_expense = month_df[month_df["type"] == "GASTO"]["amount"].sum()
    balance = total_income - total_expense
    savings_pct = savings_percentage(total_income, total_expense)
    budget_used_pct = total_expense / monthly_budget_total if monthly_budget_total else 0

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Ingresos", f"{total_income:,.0f}")
    kpi_cols[1].metric("Gastos", f"{total_expense:,.0f}", delta=f"{budget_used_pct*100:.1f}% del presupuesto")
    kpi_cols[2].metric("Balance", f"{balance:,.0f}")
    kpi_cols[3].metric("Ahorro %", f"{savings_pct*100:.1f}%")
    kpi_cols[4].metric("Uso de presupuesto", f"{budget_used_pct*100:.1f}%")

    # Alerts
    if total_expense > monthly_budget_total and monthly_budget_total > 0:
        st.error("🚨 Over budget this month")
    if config and savings_pct < config[3]:
        st.warning("⚠️ Ahorro por debajo de la meta")

    col_chart1, col_chart2 = st.columns(2)
    monthly_totals = compute_monthly_totals(df, selected_year)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=monthly_totals["month"], y=monthly_totals.get("INGRESO", 0), name="Ingresos"))
    fig_line.add_trace(go.Scatter(x=monthly_totals["month"], y=monthly_totals.get("GASTO", 0), name="Gastos"))
    fig_line.update_layout(title="Ingresos vs Gastos", xaxis_title="Mes", yaxis_title="Monto")
    col_chart1.plotly_chart(fig_line, use_container_width=True)

    category_totals = get_category_totals(df, selected_month, selected_year)
    if not category_totals.empty:
        fig_pie = px.pie(category_totals, values="amount", names="category", title="Gastos por categoría")
        col_chart2.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Resumen mensual")
    summary_rows = []
    for m in range(1, 13):
        month_data = df[(df["month"] == m) & (df["year"] == selected_year)]
        incomes = month_data[month_data["type"] == "INGRESO"]["amount"].sum()
        expenses = month_data[month_data["type"] == "GASTO"]["amount"].sum()
        balance_m = incomes - expenses
        savings_pct_m = savings_percentage(incomes, expenses)
        fixed_exp = month_data[(month_data["type"] == "GASTO") & (month_data["fixedVariable"] == "FIJO")]["amount"].sum()
        var_exp = month_data[(month_data["type"] == "GASTO") & (month_data["fixedVariable"] == "VARIABLE")]["amount"].sum()
        essential_exp = month_data[(month_data["type"] == "GASTO") & (month_data["essential"] == True)]["amount"].sum()
        non_essential = expenses - essential_exp
        budget_used = expenses / monthly_budget_total if monthly_budget_total else 0
        summary_rows.append(
            [
                month_names[m],
                incomes,
                expenses,
                balance_m,
                savings_pct_m * 100,
                fixed_exp,
                var_exp,
                (essential_exp / expenses * 100) if expenses else 0,
                (non_essential / expenses * 100) if expenses else 0,
                monthly_budget_total,
                budget_used * 100,
            ]
        )
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "Mes",
            "Ingresos",
            "Gastos",
            "Balance",
            "Ahorro %",
            "Gastos Fijos",
            "Gastos Variables",
            "% Esenciales",
            "% No esenciales",
            "Presupuesto plan",
            "Uso presupuesto %",
        ],
    )
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Gastos por categoría")
    expenses_by_cat = (
        df[df["type"] == "GASTO"]
        .groupby("category")
        .agg({"amount": "sum", "fixedVariable": "first", "essential": "first"})
        .reset_index()
    )
    total_expenses = expenses_by_cat["amount"].sum()
    expenses_by_cat["% del total"] = expenses_by_cat["amount"] / total_expenses * 100 if total_expenses else 0
    st.dataframe(expenses_by_cat, use_container_width=True)


def transactions_page(user):
    st.header("Movimientos")
    df = load_transactions_df(user["id"])
    if df.empty:
        st.info("No hay transacciones aún")
    else:
        col1, col2, col3, col4 = st.columns(4)
        start = col1.date_input("Desde", value=df["date"].min())
        end = col2.date_input("Hasta", value=df["date"].max())
        type_filter = col3.selectbox("Tipo", options=["Todos", "INGRESO", "GASTO"], index=0)
        categories = list(df["category"].unique())
        cat_filter = col4.selectbox("Categoría", options=["Todas"] + categories)

        filtered = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
        if type_filter != "Todos":
            filtered = filtered[filtered["type"] == type_filter]
        if cat_filter != "Todas":
            filtered = filtered[filtered["category"] == cat_filter]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.subheader("Agregar / Editar")
    categories_income = list_categories(user["id"], "INGRESO")
    categories_expense = list_categories(user["id"], "GASTO")
    categories_map = {c[0]: f"{c[2]} ({c[1]})" for c in categories_income + categories_expense}
    with st.form("tx_form"):
        tx_id = st.selectbox("Transacción a editar (opcional)", options=[None] + df["id"].tolist() if not df.empty else [None])
        tx_date = st.date_input("Fecha", value=date.today())
        tx_type = st.selectbox("Tipo", options=["INGRESO", "GASTO"])
        tx_categories = categories_income if tx_type == "INGRESO" else categories_expense
        cat_options = {c[0]: c[2] for c in tx_categories}
        category_id = st.selectbox("Categoría", options=list(cat_options.keys()), format_func=lambda x: cat_options[x] if x in cat_options else "")
        amount = st.number_input("Monto", min_value=0.0, step=100.0)
        payment_method = st.text_input("Método de pago", value="Santander")
        notes = st.text_area("Notas", value="")
        submitted = st.form_submit_button("Guardar")
    if submitted:
        create_or_update_transaction(user["id"], tx_id, tx_date, tx_type, category_id, amount, payment_method, notes)
        st.success("Transacción guardada")
        st.experimental_rerun()

    if not df.empty:
        with st.expander("Eliminar transacción"):
            del_id = st.selectbox("Seleccionar", options=df["id"].tolist())
            if st.button("Eliminar", type="primary"):
                delete_transaction(user["id"], del_id)
                st.success("Eliminada")
                st.experimental_rerun()

    st.subheader("Exportar CSV")
    csv_df = export_transactions_csv(user["id"])
    st.download_button("Exportar movimientos CSV", data=csv_df.to_csv(index=False).encode("utf-8"), file_name="movimientos.csv")


def categories_page(user):
    st.header("Config & Categorías")
    config = get_config(user["id"])
    default_year = config[1] if config else datetime.utcnow().year
    with st.form("config_form"):
        year = st.number_input("Año", value=default_year, step=1)
        currency = st.text_input("Moneda", value=config[2] if config else "UYU")
        savings_target = st.slider("Meta de ahorro %", 0.0, 1.0, value=config[3] if config else 0.2)
        submitted = st.form_submit_button("Guardar config")
    if submitted:
        upsert_config(user["id"], int(year), currency, float(savings_target))
        st.success("Config actualizada")

    st.subheader("Categorías")
    cats = list_categories(user["id"])
    cats_df = pd.DataFrame(cats, columns=["id", "Tipo", "Nombre", "Fijo/Variable", "Esencial", "Presupuesto mensual"])
    st.dataframe(cats_df, use_container_width=True)

    with st.form("cat_form"):
        cat_id = st.selectbox("Editar categoría", options=[None] + cats_df["id"].tolist())
        cat_type = st.selectbox("Tipo", options=["INGRESO", "GASTO"])
        name = st.text_input("Nombre")
        fixed_var = st.selectbox("Fijo/Variable", options=[None, "FIJO", "VARIABLE"], help="Solo para gastos")
        essential = st.checkbox("Esencial", value=True)
        budget = st.number_input("Presupuesto mensual", min_value=0.0, step=500.0)
        submitted_cat = st.form_submit_button("Guardar categoría")
    if submitted_cat:
        create_or_update_category(
            user["id"], cat_id, cat_type, name, fixed_var if cat_type == "GASTO" else None, essential, budget if budget > 0 else None
        )
        st.success("Categoría guardada")
        st.experimental_rerun()

    with st.expander("Eliminar categoría"):
        if not cats_df.empty:
            del_cat_id = st.selectbox("Seleccionar", options=cats_df["id"].tolist())
            if st.button("Eliminar categoría", type="primary"):
                ok = delete_category(user["id"], del_cat_id)
                if ok:
                    st.success("Eliminada")
                    st.experimental_rerun()
                else:
                    st.error("No se puede eliminar: hay transacciones asociadas")


def forecast_page(user):
    st.header("Forecast")
    df = load_transactions_df(user["id"])
    if df.empty:
        st.info("No hay datos para proyectar")
        return
    df_sorted = df.sort_values("date")
    last_date = df_sorted["date"].max()
    historical_expenses = df_sorted[df_sorted["type"] == "GASTO"]
    historical_incomes = df_sorted[df_sorted["type"] == "INGRESO"]
    exp_monthly = historical_expenses.groupby(["year", "month"]).agg({"amount": "sum"}).reset_index()
    inc_monthly = historical_incomes.groupby(["year", "month"]).agg({"amount": "sum"}).reset_index()

    def rolling_avg(df_monthly):
        if df_monthly.empty:
            return 0
        last_three = df_monthly.tail(3)["amount"]
        if last_three.empty:
            return df_monthly["amount"].mean()
        return last_three.mean()

    avg_exp = rolling_avg(exp_monthly)
    avg_inc = rolling_avg(inc_monthly)

    forecasts = []
    for i in range(1, 7):
        future_date = (last_date + pd.DateOffset(months=i)).to_pydatetime()
        bal = avg_inc - avg_exp
        savings_pct = savings_percentage(avg_inc, avg_exp) * 100
        forecasts.append(
            [
                future_date.strftime("%B %Y"),
                avg_inc,
                avg_exp,
                bal,
                savings_pct,
            ]
        )
    forecast_df = pd.DataFrame(
        forecasts,
        columns=["Mes", "Ingresos proyectados", "Gastos proyectados", "Balance", "Ahorro %"],
    )
    st.dataframe(forecast_df, use_container_width=True)

    history_vs_forecast = pd.concat(
        [
            exp_monthly.assign(label="Histórico"),
            pd.DataFrame(
                {
                    "year": [last_date.year + (last_date.month + i - 1) // 12 for i in range(1, 7)],
                    "month": [((last_date.month + i - 1) % 12) + 1 for i in range(1, 7)],
                    "amount": avg_exp,
                    "label": "Forecast",
                }
            ),
        ]
    )
    history_vs_forecast["period"] = history_vs_forecast.apply(lambda r: f"{int(r['month']):02d}/{int(r['year'])}", axis=1)
    fig = px.line(history_vs_forecast, x="period", y="amount", color="label", title="Gastos históricos vs forecast")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# App
# ----------------------------


if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    login_form()
    st.stop()

user = st.session_state.user
st.sidebar.title("💼 Finance CRM")
st.sidebar.write(user["name"])
page = st.sidebar.radio("Navegación", ["Dashboard", "Movimientos", "Categorías & Config", "Forecast"])
if st.sidebar.button("Cerrar sesión"):
    st.session_state.user = None
    st.experimental_rerun()

if page == "Dashboard":
    dashboard_page(user)
elif page == "Movimientos":
    transactions_page(user)
elif page == "Categorías & Config":
    categories_page(user)
elif page == "Forecast":
    forecast_page(user)
