from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # plotly is optional; app still runs without charts
    px = None


# =========================
# Page config + styling (skip at line 377 for Data exploration)
# =========================
st.set_page_config(
    page_title="Imobiliare AI • București",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  /* Slightly tighter default spacing */
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

  /* Header card */
  .hero {
    border-radius: 16px;
    padding: 18px 18px;
    background: radial-gradient(1200px circle at 0% 0%, rgba(99,102,241,.28), transparent 50%),
                radial-gradient(800px circle at 100% 0%, rgba(16,185,129,.22), transparent 48%),
                linear-gradient(135deg, rgba(15,23,42,.92), rgba(2,6,23,.92));
    border: 1px solid rgba(148, 163, 184, 0.18);
  }
  .hero h1 { margin: 0; font-size: 1.8rem; color: #e2e8f0; }
  .hero p { margin: .35rem 0 0 0; color: rgba(226,232,240,.85); }

  /* Small pill */
  .pill {
    display:inline-block;
    padding: 2px 10px;
    margin-right: 8px;
    border-radius: 999px;
    font-size: .8rem;
    background: rgba(148,163,184,.12);
    border: 1px solid rgba(148,163,184,.18);
    color: rgba(226,232,240,.9);
  }

  /* Result card */
  .result {
    border-radius: 16px;
    padding: 16px;
    background: linear-gradient(135deg, rgba(2,6,23,.75), rgba(15,23,42,.55));
    border: 1px solid rgba(148, 163, 184, 0.18);
  }
  .result .big { font-size: 2rem; font-weight: 700; color: #e2e8f0; margin: 0; }
  .result .muted { color: rgba(226,232,240,.75); margin: .35rem 0 0 0; }

    /* Green success result variant */
    .result--success {
        background: linear-gradient(135deg, rgba(16,185,129,.35), rgba(16,185,129,.12));
        border: 1px solid rgba(16,185,129,.55);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Domain constants
# =========================
NUMERIC_COLS: list[str] = [
    "rooms_count",
    "useful_surface",
    "built_surface",
    "construction_year",
    "bathrooms_count",
    "level",
    "max_level",
    "garages_count",
]


@dataclass(frozen=True)
class Artifacts:
    model: object
    scaler: object
    feature_columns: list[str]


def _format_eur(value: float) -> str:
    # 123,456 -> 123.456 (RO-ish) + €
    s = f"{value:,.0f}"
    s = s.replace(",", " ").replace(" ", ".")
    return f"{s} €"


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Artifacts:
    model = joblib.load("models/model_linreg.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    # Prefer models/scaler.pkl (as your UI expects), fallback to data/scaler.pkl (from preprocessing)
    scaler_path_candidates: Iterable[str] = ("models/scaler.pkl", "data/scaler.pkl")
    scaler = None
    for p in scaler_path_candidates:
        if os.path.exists(p):
            scaler = joblib.load(p)
            break
    if scaler is None:
        raise FileNotFoundError(
            "Nu găsesc scaler-ul. Aștept `models/scaler.pkl` sau `data/scaler.pkl`. "
            "Rulează `1.preprocesare.py` și/sau verifică fișierele din `models/`."
        )

    return Artifacts(model=model, scaler=scaler, feature_columns=list(feature_columns))


def build_feature_row(
    *,
    artifacts: Artifacts,
    rooms_count: int,
    useful_surface: float,
    built_surface: float,
    construction_year: int,
    bathrooms_count: int,
    level: int,
    max_level: int,
    garages_count: int,
    location_area: str | None,
) -> pd.DataFrame:
    row = {col: 0 for col in artifacts.feature_columns}

    # raw numeric values; we scale them next
    row["rooms_count"] = rooms_count
    row["useful_surface"] = useful_surface
    row["built_surface"] = built_surface
    row["construction_year"] = construction_year
    row["bathrooms_count"] = bathrooms_count
    row["level"] = level
    row["max_level"] = max_level
    row["garages_count"] = garages_count

    if location_area:
        col_name = "location_area_" + location_area
        if col_name in row:
            row[col_name] = 1

    X_input = pd.DataFrame([row], columns=artifacts.feature_columns)
    X_input[NUMERIC_COLS] = artifacts.scaler.transform(X_input[NUMERIC_COLS])
    return X_input


def predict_price_eur(artifacts: Artifacts, X_input: pd.DataFrame) -> float:
    pred_log = float(artifacts.model.predict(X_input)[0])
    return float(np.expm1(pred_log))


artifacts = load_artifacts()

# Extract locations from training columns (drop_first=True => one category is the implicit base)
location_onehot_cols = [c for c in artifacts.feature_columns if c.startswith("location_area_")]
location_names = [c.replace("location_area_", "") for c in location_onehot_cols]


# =========================
# Header
# =========================
st.markdown(
    """
<div class="hero">
  <h1>Imobiliare AI • Estimator de preț</h1>
  <p>Introdu detalii despre locuință și primești o estimare rapidă (în €). Include și predicții batch + explorare de date.</p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Sidebar navigation
# =========================
with st.sidebar:
    st.subheader("Navigație")
    page = st.radio(
        "",
        options=["Estimator", "Predicții batch", "Explorare date", "Despre"],
        index=0,
    )

    st.divider()
    st.caption("Artefacte încărcate")
    st.write("- `models/model_linreg.pkl`")
    st.write("- `models/feature_columns.pkl`")
    st.write("- scaler: `models/scaler.pkl` (fallback `data/scaler.pkl`)")


# =========================
# Page: Estimator
# =========================
if page == "Estimator":
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Introdu datele locuinței")
        with st.form("predict_form", border=True):
            c1, c2 = st.columns(2)

            with c1:
                rooms_count = st.number_input("Număr camere", min_value=0, max_value=20, value=3, step=1)
                useful_surface = st.number_input("Suprafață utilă (mp)", min_value=1.0, value=50.0, step=10.0)
                construction_year = st.number_input(
                    "An construcție",
                    min_value=1700,
                    max_value=2026,
                    value=2010,
                    step=1,
                )
                bathrooms_count = st.number_input("Număr băi", min_value=0, max_value=10, value=1, step=1)

            with c2:
                built_surface = st.number_input("Suprafață construită (mp)", min_value=0.0, value=70.0, step=10.0)
                level = st.number_input("Etaj", min_value=0, max_value=60, value=2, step=1)
                max_level = st.number_input("Etaje bloc", min_value=0, max_value=200, value=8, step=1)
                garages_count = st.number_input("Număr garaje", min_value=0, max_value=10, value=0, step=1)

            st.divider()
            st.caption("Zonă / cartier (opțional)")
            location_choice = st.selectbox(
                "Cartier / zonă",
                options=["(fără selecție — baza drop_first)"] + sorted(location_names),
                index=0,
            )

            submitted = st.form_submit_button("Estimează prețul", use_container_width=True)

    with right:
        st.subheader("Rezultat")
        if submitted:
            location_area = None if location_choice.startswith("(") else location_choice

            if max_level > 0 and level > max_level:
                st.warning("Etajul este mai mare decât numărul de etaje ale blocului. Verifică valorile.")

            X_input = build_feature_row(
                artifacts=artifacts,
                rooms_count=int(rooms_count),
                useful_surface=float(useful_surface),
                built_surface=float(built_surface),
                construction_year=int(construction_year),
                bathrooms_count=int(bathrooms_count),
                level=int(level),
                max_level=int(max_level),
                garages_count=int(garages_count),
                location_area=location_area,
            )
            price = predict_price_eur(artifacts, X_input)
            price_per_mp = price / max(float(useful_surface), 1.0)

            st.markdown(
                f"""
<div class="result result--success">
  <p class="big">{_format_eur(price)}</p>
  <p class="muted">≈ {_format_eur(price_per_mp)} / mp (după suprafața utilă)</p>
</div>
""",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Camere", int(rooms_count))
            m2.metric("Suprafață utilă", f"{float(useful_surface):.0f} mp")
            m3.metric("Zonă", location_area or "(baza drop_first)")

            st.caption("Modelul prezice `price_log`, apoi convertește înapoi cu `expm1()`.")
        else:
            st.info("Completează formularul și apasă «Estimează prețul».", icon="ℹ️")


# =========================
# Page: Batch predictions
# =========================
elif page == "Predicții batch":
    st.subheader("Predicții batch (CSV upload)")
    st.write(
        "Încarcă un CSV cu coloanele: `rooms_count`, `useful_surface`, `built_surface`, `construction_year`, "
        "`bathrooms_count`, `level`, `max_level`, `garages_count` și opțional `location_area`."
    )

    sample = pd.DataFrame(
        [
            {
                "rooms_count": 2,
                "useful_surface": 52,
                "built_surface": 70,
                "construction_year": 2012,
                "bathrooms_count": 1,
                "level": 3,
                "max_level": 8,
                "garages_count": 0,
                "location_area": "Titan",
            }
        ]
    )
    st.download_button(
        "Descarcă un exemplu CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="exemplu_pred.csv",
        mime="text/csv",
        use_container_width=True,
    )

    upload = st.file_uploader("CSV", type=["csv"], accept_multiple_files=False)
    if upload is not None:
        df = pd.read_csv(upload)
        missing = [c for c in NUMERIC_COLS if c not in df.columns]
        if missing:
            st.error(f"Lipsesc coloanele obligatorii: {missing}")
            st.stop()

        df = df.copy()
        if "location_area" not in df.columns:
            df["location_area"] = None

        # Build feature matrix aligned to training columns
        rows = []
        for _, r in df.iterrows():
            loc = r.get("location_area")
            loc = None if pd.isna(loc) else str(loc).strip()
            rows.append(
                build_feature_row(
                    artifacts=artifacts,
                    rooms_count=int(r["rooms_count"]),
                    useful_surface=float(r["useful_surface"]),
                    built_surface=float(r["built_surface"]),
                    construction_year=int(r["construction_year"]),
                    bathrooms_count=int(r["bathrooms_count"]),
                    level=int(r["level"]),
                    max_level=int(r["max_level"]),
                    garages_count=int(r["garages_count"]),
                    location_area=loc,
                )
            )
        X = pd.concat(rows, ignore_index=True)
        preds = artifacts.model.predict(X)
        df_out = df.copy()
        df_out["pred_price_eur"] = np.expm1(preds)
        st.dataframe(df_out, use_container_width=True, hide_index=True)

        st.download_button(
            "Descarcă predicțiile",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictii.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================
# Page: Data exploration
# =========================
elif page == "Explorare date":
    st.subheader("Explorare rapidă (dataset brut)")
    st.caption("Citește din `data/house_offers.csv` dacă există.")

    df_raw = _safe_read_csv("data/house_offers.csv")
    if df_raw is None:
        st.warning("Nu găsesc `data/house_offers.csv`.", icon="⚠️")
        st.stop()

    keep = ["price", "location_area", "useful_surface", "construction_year"]
    dfv = df_raw[keep].copy()
    dfv["price"] = pd.to_numeric(dfv["price"], errors="coerce")
    dfv["useful_surface"] = pd.to_numeric(dfv["useful_surface"], errors="coerce")
    dfv["construction_year"] = pd.to_numeric(dfv["construction_year"], errors="coerce")
    dfv["location_area"] = dfv["location_area"].astype(str).str.strip()
    dfv = dfv.dropna()
    dfv = dfv[(dfv["price"] > 0) & (dfv["useful_surface"] > 0)]
    dfv["price_per_mp"] = dfv["price"] / dfv["useful_surface"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Anunțuri", f"{len(dfv):,}".replace(",", "."))
    c2.metric("Preț median", _format_eur(float(dfv["price"].median())))
    c3.metric("€/mp median", _format_eur(float(dfv["price_per_mp"].median())))

    st.divider()
    top_n = st.slider("Top N cartiere (€/mp median)", min_value=5, max_value=30, value=10)
    by_area = (
        dfv.groupby("location_area", as_index=False)
        .agg(median_eur_mp=("price_per_mp", "median"), count=("price", "size"))
        .sort_values("median_eur_mp", ascending=False)
        .head(top_n)
    )

    if px is not None:
        fig = px.bar(
            by_area.sort_values("median_eur_mp", ascending=True),
            x="median_eur_mp",
            y="location_area",
            orientation="h",
            title=f"Top {top_n} cartiere după €/mp (median)",
            hover_data={"count": True, "median_eur_mp": ":.0f"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Instalează `plotly` pentru grafice interactive (opțional).")
        st.dataframe(by_area, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Preview date")
    st.dataframe(dfv.sample(min(200, len(dfv))), use_container_width=True, hide_index=True)


# =========================
# Page: About
# =========================
else:
    st.subheader("Despre")
    st.write(
        "Aplicația folosește un model de regresie liniară antrenat pe dataset-ul preprocesat "
        "(`data/bucuresti_ready.csv`). Interfața aliniază input-ul la `feature_columns.pkl` și aplică "
        "același `StandardScaler` ca în preprocessing."
    )
    st.markdown(
        """
**Tips**
- Dacă ai schimbat preprocessing-ul (coloane / dummies), re-rulează `1.preprocesare.py` și `3si4.antrenare&evaluare.py`.
- Pentru UI: `venv\\Scripts\\python -m streamlit run 5.interfata.py`.
"""
    )
