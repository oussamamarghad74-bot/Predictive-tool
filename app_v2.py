import streamlit as st
import pandas as pd
import plotly.express as px


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# Global Settings – Gabelstapler Predictive Maintenance
# =========================================================

# حالات الرافعة الشوكية
CLASS_ORDER = ["Gut", "Warnung", "Kritisch", "Ausfall"]

# وقت الصيانة المتبقي بالساعات
RUL_BASE = {
    "Gut": 120,
    "Warnung": 48,
    "Kritisch": 12,
    "Ausfall": 2
}

SEVERITY = {
    "Gut": 0,
    "Warnung": 1,
    "Kritisch": 2,
    "Ausfall": 3
}

STATE_COLORS = {
    "Gut": "#16a34a",
    "Warnung": "#f59e0b",
    "Kritisch": "#ef4444",
    "Ausfall": "#7f1d1d"
}

DECISION_COLORS = {
    "MONITORING": "#2563eb",
    "VORWARNUNG": "#7c3aed",
    "WARTUNGSAUFTRAG": "#16a34a",
    "TECHNIKER_FREIGABE": "#f59e0b",
    "UNSICHER_WARNUNG": "#fb7185",
    "SOFORT_STOPP": "#dc2626",
    "TEILE_FEHLEN": "#b45309"
}

# أنواع المكونات التي نراقبها
KOMPONENTEN = ["Batterie", "Motor", "Reifen", "Hydraulik", "Bremsen"]

# ألوان المكونات
KOMPONENTEN_COLORS = {
    "Batterie": "#38bdf8",
    "Motor": "#a855f7",
    "Reifen": "#f59e0b",
    "Hydraulik": "#22c55e",
    "Bremsen": "#ef4444"
}

# حدود الصيانة بالساعات
WARTUNG_GRENZEN = {
    "Batterie": {"kritisch": 10, "warnung": 30, "max": 100},
    "Motor": {"kritisch": 20, "warnung": 60, "max": 300},
    "Reifen": {"kritisch": 15, "warnung": 50, "max": 200},
    "Hydraulik": {"kritisch": 25, "warnung": 80, "max": 250},
    "Bremsen": {"kritisch": 10, "warnung": 35, "max": 150}
}

# =========================================================
# Factory Info – LogisTech GmbH
# =========================================================

FACTORY_INFO = {
    "name": "LogisTech GmbH",
    "werk": "Werk 1 – Hamburg",
    "adresse": "Hafenstraße 23, 20457 Hamburg",
    "gegruendet": "2010",
    "mitarbeiter": 284,
    "zertifikate": ["ISO 9001:2015", "ISO 45001", "VDI 2198"],
    "flotte": 24,
    "schichten": ["Frühschicht 06:00–14:00", "Spätschicht 14:00–22:00", "Nachtschicht 22:00–06:00"]
}

# =========================================================
# Gabelstapler Registry
# =========================================================

GABELSTAPLER_REGISTRY = {
    "G01": {
        "name": "Linde E25L",
        "typ": "Elektro-Gegengewichtsstapler",
        "bereich": "Halle A – Eingang",
        "bediener": "K. Weber",
        "sensor_id": "SNS-2024-G01-VIB",
        "baujahr": 2020,
        "tragkraft_kg": 2500,
        "batterietyp": "Li-Ion 48V"
    },
    "G02": {
        "name": "Toyota 8FBE20",
        "typ": "Elektro-Gegengewichtsstapler",
        "bereich": "Halle A – Ausgang",
        "bediener": "T. Müller",
        "sensor_id": "SNS-2024-G02-VIB",
        "baujahr": 2019,
        "tragkraft_kg": 2000,
        "batterietyp": "Blei-Säure 48V"
    },
    "G03": {
        "name": "Still RX20-20",
        "typ": "Elektro-Schubmaststapler",
        "bereich": "Halle B – Lager",
        "bediener": "F. Schmidt",
        "sensor_id": "SNS-2024-G03-VIB",
        "baujahr": 2021,
        "tragkraft_kg": 2000,
        "batterietyp": "Li-Ion 48V"
    },
    "G04": {
        "name": "Jungheinrich EFG425",
        "typ": "Elektro-Gegengewichtsstapler",
        "bereich": "Halle B – Produktion",
        "bediener": "R. Bauer",
        "sensor_id": "SNS-2024-G04-VIB",
        "baujahr": 2018,
        "tragkraft_kg": 2500,
        "batterietyp": "Blei-Säure 48V"
    },
    "G05": {
        "name": "Linde H25D",
        "typ": "Diesel-Gegengewichtsstapler",
        "bereich": "Außengelände",
        "bediener": "M. Klein",
        "sensor_id": "SNS-2024-G05-VIB",
        "baujahr": 2019,
        "tragkraft_kg": 2500,
        "batterietyp": "Diesel"
    },
    "G06": {
        "name": "Crown RC5500",
        "typ": "Elektro-Schubmaststapler",
        "bereich": "Halle C – Hochregal",
        "bediener": "S. Hoffmann",
        "sensor_id": "SNS-2024-G06-VIB",
        "baujahr": 2022,
        "tragkraft_kg": 1800,
        "batterietyp": "Li-Ion 48V"
    },
    "G07": {
        "name": "Hyster H2.0FT",
        "typ": "Treibgas-Gegengewichtsstapler",
        "bereich": "Halle A – Versand",
        "bediener": "A. Fischer",
        "sensor_id": "SNS-2024-G07-VIB",
        "baujahr": 2020,
        "tragkraft_kg": 2000,
        "batterietyp": "LPG Gas"
    },
    "G08": {
        "name": "Komatsu FB20M",
        "typ": "Elektro-Gegengewichtsstapler",
        "bereich": "Halle C – Kommissionierung",
        "bediener": "P. Wagner",
        "sensor_id": "SNS-2024-G08-VIB",
        "baujahr": 2021,
        "tragkraft_kg": 2000,
        "batterietyp": "Li-Ion 48V"
    },
    "G09": {
        "name": "BT Reflex RRE160",
        "typ": "Elektro-Schubmaststapler",
        "bereich": "Halle B – Hochregal",
        "bediener": "L. Braun",
        "sensor_id": "SNS-2024-G09-VIB",
        "baujahr": 2020,
        "tragkraft_kg": 1600,
        "batterietyp": "Li-Ion 48V"
    },
    "G10": {
        "name": "Manitou MI25G",
        "typ": "Treibgas-Gelenkmaststapler",
        "bereich": "Außengelände – Süd",
        "bediener": "J. Schulz",
        "sensor_id": "SNS-2024-G10-VIB",
        "baujahr": 2018,
        "tragkraft_kg": 2500,
        "batterietyp": "LPG Gas"
    },
    "G11": {
        "name": "Doosan B20X-7",
        "typ": "Elektro-Gegengewichtsstapler",
        "bereich": "Halle D – Eingang",
        "bediener": "C. Richter",
        "sensor_id": "SNS-2024-G11-VIB",
        "baujahr": 2022,
        "tragkraft_kg": 2000,
        "batterietyp": "Li-Ion 48V"
    },
    "G12": {
        "name": "Linde T20AP",
        "typ": "Elektro-Deichselstapler",
        "bereich": "Halle D – Kommissionierung",
        "bediener": "H. Wolf",
        "sensor_id": "SNS-2024-G12-VIB",
        "baujahr": 2021,
        "tragkraft_kg": 2000,
        "batterietyp": "Li-Ion 24V"
    },
}

# =========================================================
# Sensor Reading für Gabelstapler
# =========================================================

def get_sensor_reading(stapler_id, zustand, betriebsstunden, seed=0):
    rng = np.random.default_rng(seed)

    base_vib = {"Gut": 0.8, "Warnung": 2.1, "Kritisch": 4.5, "Ausfall": 7.2}
    base_temp = {"Gut": 45, "Warnung": 62, "Kritisch": 78, "Ausfall": 95}
    base_batterie = {"Gut": 92, "Warnung": 71, "Kritisch": 45, "Ausfall": 18}
    base_hydraulik = {"Gut": 180, "Warnung": 145, "Kritisch": 110, "Ausfall": 75}
    base_strom = {"Gut": 85, "Warnung": 110, "Kritisch": 145, "Ausfall": 180}

    stapler_info = GABELSTAPLER_REGISTRY.get(stapler_id, {})

    return {
        "sensor_id": stapler_info.get("sensor_id", "SNS-UNKNOWN"),
        "timestamp": pd.Timestamp.now().strftime("%d.%m.%Y – %H:%M:%S"),
        "messung_nr": int(rng.integers(4000, 9999)),
        "vibration_mm_s": round(base_vib[zustand] + rng.normal(0, 0.2), 2),
        "motor_temp_c": round(base_temp[zustand] + rng.normal(0, 2.5), 1),
        "batterie_pct": round(base_batterie[zustand] + rng.normal(0, 3), 1),
        "hydraulikdruck_bar": round(base_hydraulik[zustand] + rng.normal(0, 5), 1),
        "motorstrom_a": round(base_strom[zustand] + rng.normal(0, 4), 1),
        "betriebsstunden": betriebsstunden,
        "ladezyklen": int(rng.integers(50, 800))
    }

def get_current_shift():
    hour = pd.Timestamp.now().hour
    if 6 <= hour < 14:
        return "Frühschicht 06:00–14:00"
    elif 14 <= hour < 22:
        return "Spätschicht 14:00–22:00"
    else:
        return "Nachtschicht 22:00–06:00"


# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Predictive Tool Logistics – Advanced Prototype",
    page_icon="🏭",
    layout="wide"
)
import time

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "update_counter" not in st.session_state:
    st.session_state.update_counter = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if time.time() - st.session_state.last_update > 5:
    st.session_state.last_update = time.time()
    st.session_state.update_counter += 1
    st.rerun()

# =========================================================
# Custom CSS Design
# =========================================================

st.markdown("""
<style>
    @keyframes slideInFromTop {
        0% { transform: translateY(-30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }

    @keyframes pulseRed {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
        50% { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
    }

    @keyframes pulseOrange {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245,158,11,0.5); }
        50% { box-shadow: 0 0 0 14px rgba(245,158,11,0); }
    }

    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
        50% { box-shadow: 0 0 0 14px rgba(16,185,129,0); }
    }

    @keyframes pulseBlue {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
        50% { box-shadow: 0 0 0 14px rgba(99,102,241,0); }
    }

    .notif-sofort {
        animation: slideInFromTop 0.5s ease-out forwards,
                   pulseRed 2s infinite 0.5s;
    }

    .notif-bestand {
        animation: slideInFromTop 0.5s ease-out forwards,
                   pulseOrange 2s infinite 0.5s;
    }

    .notif-auto {
        animation: slideInFromTop 0.6s ease-out forwards,
                   pulseGreen 2.5s infinite 0.6s;
    }

    .notif-vorwarnung {
        animation: slideInFromTop 0.7s ease-out forwards,
                   pulseBlue 3s infinite 0.7s;
    }

    .main {
        background-color: #0f172a;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    p, label, div {
        color: #e5e7eb;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #e5e7eb;
        border-radius: 12px;
        padding: 12px 18px;
        border: 1px solid #334155;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
    }

    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 55%, #111827 100%);
        padding: 28px;
        border-radius: 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #dbeafe;
    }

    .kpi-card {
        background: linear-gradient(145deg, #111827, #1e293b);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        min-height: 118px;
    }

    .kpi-title {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 31px;
        font-weight: 800;
        color: #f8fafc;
    }

    .kpi-sub {
        font-size: 13px;
        color: #cbd5e1;
        margin-top: 6px;
    }

    .section-card {
        background-color: #111827;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.22);
    }

    .badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 13px;
    }

    .stDataFrame {
        background-color: #111827;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# Helper UI Functions
# =========================================================

def kpi_card(title, value, subtitle="", color="#38bdf8"):
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 4px solid {color};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def badge(text, color):
    return f"""
    <span class="badge" style="background-color:{color}; color:white;">
        {text}
    </span>
    """



# =========================================================
# Gabelstapler Fleet Generation
# =========================================================

def build_gabelstapler_fleet(n_stapler=12, scenario="Normalbetrieb", seed=123):
    rng = np.random.default_rng(seed)

    bereiche = [
        "Halle A – Eingang", "Halle A – Ausgang",
        "Halle B – Lager", "Halle B – Produktion",
        "Halle C – Hochregal", "Halle C – Kommissionierung",
        "Halle D – Eingang", "Halle D – Kommissionierung",
        "Außengelände – Nord", "Außengelände – Süd"
    ]

    typen = [
        "Elektro-Gegengewichtsstapler",
        "Elektro-Schubmaststapler",
        "Diesel-Gegengewichtsstapler",
        "Treibgas-Gelenkmaststapler",
        "Elektro-Deichselstapler"
    ]

    probs_map = {
        "Normalbetrieb": [0.45, 0.30, 0.17, 0.08],
        "Wartungskrise": [0.20, 0.30, 0.32, 0.18],
        "Schichtende": [0.30, 0.35, 0.25, 0.10],
        "Hochbetrieb": [0.35, 0.32, 0.22, 0.11]
    }

    probs = probs_map.get(scenario, probs_map["Normalbetrieb"])

    rows = []
    stapler_ids = list(GABELSTAPLER_REGISTRY.keys())

    for i in range(min(n_stapler, len(stapler_ids))):
        g_id = stapler_ids[i]
        g_info = GABELSTAPLER_REGISTRY[g_id]

        x = 1.5 + (i % 4) * 3.0 + rng.normal(0, 0.15)
        y = 1.5 + (i // 4) * 3.0 + rng.normal(0, 0.15)

        zustand = rng.choice(CLASS_ORDER, p=probs)
        betriebsstunden = int(rng.integers(500, 8000))

        rows.append({
            "Stapler": g_id,
            "Name": g_info["name"],
            "Typ": g_info["typ"],
            "Bereich": g_info["bereich"],
            "Bediener": g_info["bediener"],
            "Sensor_ID": g_info["sensor_id"],
            "Baujahr": g_info["baujahr"],
            "Tragkraft_kg": g_info["tragkraft_kg"],
            "Batterietyp": g_info["batterietyp"],
            "X": x,
            "Y": y,
            "Ist_Zustand": zustand,
            "Betriebsstunden": betriebsstunden,
            "Stillstandskosten_EUR_h": int(rng.integers(50, 180)),
            "Fällig_in_h": int(rng.integers(5, 80)),
            "Auftrag": f"LOG-{rng.integers(1000, 9999)}"
        })

    return pd.DataFrame(rows)


# =========================================================
# RUL Schätzung für Komponenten
# =========================================================

def estimate_komponenten_rul(zustand, betriebsstunden, seed=1):
    rng = np.random.default_rng(seed)

    komponenten_rul = {}

    for komp in KOMPONENTEN:
        base = RUL_BASE[zustand]
        grenzen = WARTUNG_GRENZEN[komp]

        faktor = {
            "Batterie": 0.8,
            "Motor": 1.2,
            "Reifen": 1.0,
            "Hydraulik": 1.3,
            "Bremsen": 0.9
        }[komp]

        noise = rng.normal(0, base * 0.1)
        rul = max(1, base * faktor + noise)

        if betriebsstunden > 5000:
            rul *= 0.85
        elif betriebsstunden > 3000:
            rul *= 0.95

        komponenten_rul[komp] = round(float(rul), 1)

    return komponenten_rul


# =========================================================
# Wartungsentscheidung
# =========================================================

def make_wartungs_decision(rul_min, vorlaufzeit, confidence,
                           zustand, teile_ok,
                           auto_threshold, manual_threshold):

    if not teile_ok and rul_min <= vorlaufzeit:
        return "TEILE_FEHLEN"

    if zustand == "Ausfall" and confidence >= 0.75 and rul_min <= vorlaufzeit:
        return "SOFORT_STOPP"

    if rul_min <= vorlaufzeit:
        if confidence >= auto_threshold:
            return "WARTUNGSAUFTRAG"
        elif confidence >= manual_threshold:
            return "TECHNIKER_FREIGABE"
        else:
            return "UNSICHER_WARNUNG"

    if rul_min <= vorlaufzeit + 8:
        return "VORWARNUNG"

    return "MONITORING"


# =========================================================
# Wartungsvorlaufzeit Berechnung
# =========================================================

def calculate_wartung_vorlaufzeit(row, sicherheitsmarge,
                                   techniker_queue, seed=0):
    rng = np.random.default_rng(seed)

    komplexitaet = {
        "Elektro-Gegengewichtsstapler": 2,
        "Elektro-Schubmaststapler": 3,
        "Diesel-Gegengewichtsstapler": 2,
        "Treibgas-Gelenkmaststapler": 2,
        "Elektro-Deichselstapler": 1
    }.get(row["Typ"], 2)

    teile_ok = rng.random() > 0.12

    diagnose = 1.5 + rng.uniform(0, 1.0)
    teile_holen = 2.0 + komplexitaet + rng.uniform(0, 2.0)
    techniker_warten = techniker_queue * 1.5 + rng.uniform(0, 2.0)
    wartung_selbst = 3.0 + komplexitaet * 2 + rng.uniform(0, 3.0)
    teile_verzoegerung = rng.integers(4, 12) if not teile_ok else 0

    total = (diagnose + teile_holen + techniker_warten +
             wartung_selbst + sicherheitsmarge + teile_verzoegerung)

    return {
        "Diagnose_h": round(float(diagnose), 1),
        "Teile_holen_h": round(float(teile_holen), 1),
        "Techniker_Warten_h": round(float(techniker_warten), 1),
        "Wartung_h": round(float(wartung_selbst), 1),
        "Sicherheitsmarge_h": round(float(sicherheitsmarge), 1),
        "Teile_Verzoegerung_h": round(float(teile_verzoegerung), 1),
        "Gesamte_Vorlaufzeit_h": round(float(total), 1),
        "Teile_OK": bool(teile_ok)
    }


# =========================================================
# KI Klassifikation (vereinfacht ohne Audio)
# =========================================================

def classify_stapler_zustand(row, global_noise, seed=0):
    rng = np.random.default_rng(seed)

    sensor = get_sensor_reading(
        stapler_id=row["Stapler"],
        zustand=row["Ist_Zustand"],
        betriebsstunden=row["Betriebsstunden"],
        seed=seed
    )

    vib = sensor["vibration_mm_s"]
    temp = sensor["motor_temp_c"]
    batt = sensor["batterie_pct"]

    noise = rng.normal(0, global_noise * 10)

    score = (
        (vib / 7.2) * 0.35 +
        (temp / 95) * 0.30 +
        ((100 - batt) / 100) * 0.35
    ) + noise * 0.1

    score = np.clip(score, 0, 1)

    if score < 0.25:
        pred = "Gut"
        confidence = round(0.85 + rng.uniform(0, 0.12), 3)
    elif score < 0.50:
        pred = "Warnung"
        confidence = round(0.75 + rng.uniform(0, 0.15), 3)
    elif score < 0.75:
        pred = "Kritisch"
        confidence = round(0.70 + rng.uniform(0, 0.18), 3)
    else:
        pred = "Ausfall"
        confidence = round(0.80 + rng.uniform(0, 0.15), 3)

    return pred, min(confidence, 0.99)


# =========================================================
# Fleet Evaluation
# =========================================================

def evaluate_gabelstapler_fleet(fleet_df, global_noise,
                                 sicherheitsmarge, techniker_queue,
                                 auto_threshold, manual_threshold, seed=999):
    rows = []

    for idx, row in fleet_df.iterrows():
        pred, confidence = classify_stapler_zustand(
            row, global_noise, seed=seed + idx * 13
        )

        komponenten_rul = estimate_komponenten_rul(
            pred, row["Betriebsstunden"], seed=seed + idx
        )

        rul_min = min(komponenten_rul.values())

        wartung = calculate_wartung_vorlaufzeit(
            row, sicherheitsmarge, techniker_queue,
            seed=seed + idx * 7
        )

        vorlaufzeit = wartung["Gesamte_Vorlaufzeit_h"]

        decision = make_wartungs_decision(
            rul_min, vorlaufzeit, confidence,
            pred, wartung["Teile_OK"],
            auto_threshold, manual_threshold
        )

        urgency_gap = vorlaufzeit - rul_min
        risk_score = (
            max(0, urgency_gap) * (row["Stillstandskosten_EUR_h"] / 50)
            + SEVERITY[pred] * 20
            + (0 if wartung["Teile_OK"] else 25)
            + (15 if row["Fällig_in_h"] < 10 else 0)
        )

        new_row = row.to_dict()
        new_row.update({
            "KI_Zustand": pred,
            "Confidence": round(confidence, 3),
            "RUL_min_h": rul_min,
            "Entscheidung": decision,
            "Risk_Score": round(float(risk_score), 1),
        })
        new_row.update(komponenten_rul)
        new_row.update(wartung)

        rows.append(new_row)

    df = pd.DataFrame(rows)

    priority_map = {
        "SOFORT_STOPP": 1,
        "TEILE_FEHLEN": 2,
        "WARTUNGSAUFTRAG": 3,
        "TECHNIKER_FREIGABE": 4,
        "UNSICHER_WARNUNG": 5,
        "VORWARNUNG": 6,
        "MONITORING": 7
    }

    df["Priorität_Rang"] = df["Entscheidung"].map(priority_map)
    return df.sort_values(
        ["Priorität_Rang", "Risk_Score"],
        ascending=[True, False]
    ).reset_index(drop=True)


# =========================================================
# Domino Effect für Gabelstapler
# =========================================================

def calculate_domino_effect(fleet_df, failed_stapler):
    failed = fleet_df[fleet_df["Stapler"] == failed_stapler].iloc[0]
    affected = []

    for _, row in fleet_df.iterrows():
        if row["Stapler"] == failed_stapler:
            continue

        impact_score = 0
        reasons = []

        if row["Bereich"] == failed["Bereich"]:
            impact_score += 40
            reasons.append("Gleicher Bereich")

        if row["Typ"] == failed["Typ"]:
            impact_score += 20
            reasons.append("Gleicher Staplertyp")

        if row["RUL_min_h"] < 10:
            impact_score += 25
            reasons.append("Kritische RUL")

        if row["Stillstandskosten_EUR_h"] > 100:
            impact_score += 15
            reasons.append("Hohe Stillstandskosten")

        if impact_score > 0:
            affected.append({
                "Stapler": row["Stapler"],
                "Name": row["Name"],
                "Bereich": row["Bereich"],
                "KI_Zustand": row["KI_Zustand"],
                "RUL_min_h": row["RUL_min_h"],
                "Impact_Score": impact_score,
                "Gründe": " | ".join(reasons),
                "Stillstandskosten_EUR_h": row["Stillstandskosten_EUR_h"]
            })

    return pd.DataFrame(affected).sort_values(
        "Impact_Score", ascending=False
    ).reset_index(drop=True)


# =========================================================
# Notifications
# =========================================================

def show_notifications(fleet_df):
    sofort = fleet_df[fleet_df["Entscheidung"] == "SOFORT_STOPP"]
    wartung = fleet_df[fleet_df["Entscheidung"] == "WARTUNGSAUFTRAG"]
    teile = fleet_df[fleet_df["Entscheidung"] == "TEILE_FEHLEN"]

    for _, row in sofort.iterrows():
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#7f1d1d,#dc2626);
                    border:2px solid #ef4444;border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:20px;">🚨</span>
                <span style="color:white;font-weight:800;font-size:16px;margin-left:8px;">
                    SOFORT-STOPP: {row['Stapler']} – {row['Name']}
                </span>
            </div>
            <div style="color:#fca5a5;font-size:13px;">
                ⏱️ Noch {row['RUL_min_h']} h | {row['Bereich']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    for _, row in teile.iterrows():
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#78350f,#b45309);
                    border:2px solid #f59e0b;border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:20px;">⚠️</span>
                <span style="color:white;font-weight:800;font-size:16px;margin-left:8px;">
                    TEILE FEHLEN: {row['Stapler']} – {row['Name']}
                </span>
            </div>
            <div style="color:#fde68a;font-size:13px;">
                ⏱️ Noch {row['RUL_min_h']} h
            </div>
        </div>
        """, unsafe_allow_html=True)

    for _, row in wartung.iterrows():
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#064e3b,#065f46);
                    border:2px solid #10b981;border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:20px;">🔧</span>
                <span style="color:white;font-weight:800;font-size:16px;margin-left:8px;">
                    WARTUNGSAUFTRAG: {row['Stapler']} – {row['Name']}
                </span>
            </div>
            <div style="color:#6ee7b7;font-size:13px;">
                ⏱️ Noch {row['RUL_min_h']} h | Vorlaufzeit: {row['Gesamte_Vorlaufzeit_h']} h
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(sofort) == 0 and len(wartung) == 0 and len(teile) == 0:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#064e3b,#065f46);
                    border:1px solid #10b981;border-radius:12px;
                    padding:12px 18px;margin-bottom:8px;">
            <span style="font-size:16px;">✅</span>
            <span style="color:#6ee7b7;font-weight:600;margin-left:8px;">
                Alle Gabelstapler normal – Keine kritischen Alarme
            </span>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# KPI Simulation
# =========================================================

def simulate_kpis(n_events=180, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for method in ["Traditionell", "Predictive Maintenance"]:
        for _ in range(n_events):
            wartungszeit = rng.uniform(3, 12)
            stillstandskosten = rng.uniform(50, 180)

            if method == "Traditionell":
                erkennungszeit = rng.uniform(0, 2)
                notfall_prob = 0.68
                ausfallrisiko = np.clip(rng.normal(0.35, 0.10), 0, 1)
                nutzung = np.clip(rng.normal(0.65, 0.09), 0, 1)
            else:
                erkennungszeit = rng.uniform(8, 40)
                notfall_prob = 0.15
                ausfallrisiko = np.clip(rng.normal(0.08, 0.04), 0, 1)
                nutzung = np.clip(rng.normal(0.88, 0.05), 0, 1)

            stillstand = max(0, wartungszeit - erkennungszeit)
            notfall = 1 if rng.random() < notfall_prob and stillstand > 0 else 0

            rows.append({
                "Methode": method,
                "Stillstand_h": stillstand,
                "Stillstandskosten_EUR": stillstand * stillstandskosten,
                "Notfallwartung": notfall,
                "Rechtzeitig": 1 if stillstand == 0 else 0,
                "Ausfallrisiko": ausfallrisiko,
                "Staplernutzung": nutzung
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("Methode").agg(
        Gesamtstillstand_h=("Stillstand_h", "sum"),
        Stillstandskosten_EUR=("Stillstandskosten_EUR", "sum"),
        Notfallwartungen=("Notfallwartung", "sum"),
        Rechtzeitige_Wartung=("Rechtzeitig", "mean"),
        Ausfallrisiko=("Ausfallrisiko", "mean"),
        Staplernutzung=("Staplernutzung", "mean")
    ).reset_index()

    return df, summary


# =========================================================
# Gauge Chart
# =========================================================

def make_gauge(title, value, min_val, max_val, unit, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "white", "size": 14}},
        number={"suffix": unit, "font": {"color": "white", "size": 20}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "white"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [min_val, max_val*0.5], "color": "#064e3b"},
                {"range": [max_val*0.5, max_val*0.75], "color": "#78350f"},
                {"range": [max_val*0.75, max_val], "color": "#7f1d1d"}
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.75,
                "value": max_val * 0.75
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        font=dict(color="white"),
        height=200,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


# =========================================================
# Factory Map für Gabelstapler
# =========================================================

def factory_map(df):
    color_values = df["Entscheidung"].map(DECISION_COLORS)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["X"], y=df["Y"],
        mode="markers+text",
        text=df["Stapler"],
        textposition="top center",
        marker=dict(
            size=np.clip(df["Risk_Score"] * 1.2 + 18, 18, 60),
            color=color_values,
            line=dict(color="white", width=1.5),
            opacity=0.92
        ),
        customdata=np.stack([
            df["Bereich"], df["KI_Zustand"],
            df["RUL_min_h"], df["Gesamte_Vorlaufzeit_h"],
            df["Entscheidung"], df["Name"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{text} – %{customdata[5]}</b><br>"
            "Bereich: %{customdata[0]}<br>"
            "KI-Zustand: %{customdata[1]}<br>"
            "RUL: %{customdata[2]} h<br>"
            "Vorlaufzeit: %{customdata[3]} h<br>"
            "Entscheidung: %{customdata[4]}<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        text=["Werkstatt"],
        textposition="bottom center",
        marker=dict(size=32, color="#22c55e", symbol="square"),
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Digitale Lagerkarte – LogisTech GmbH Werk 1",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=520,
        xaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False, title="Layout X"),
        yaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False, title="Layout Y"),
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig

# =========================================================
# Header
# =========================================================
current_shift = get_current_shift()
now_str = pd.Timestamp.now().strftime("%d.%m.%Y – %H:%M:%S")

st.markdown(f"""
<div class="hero">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <div style="font-size:13px;color:#93c5fd;font-weight:600;
                        letter-spacing:2px;margin-bottom:6px;">
                ⚙️ INDUSTRIE 4.0 – PREDICTIVE MAINTENANCE
            </div>
            <div style="font-size:38px;font-weight:800;color:white;margin-bottom:5px;">
                🚜 LogisTech GmbH – Werk 1
            </div>
            <div style="font-size:18px;color:#dbeafe;">
                Predictive Gabelstapler Maintenance System | Hamburg
            </div>
            <div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;">
                <span style="background:rgba(34,197,94,0.2);border:1px solid #22c55e;
                             padding:4px 12px;border-radius:999px;font-size:12px;color:#22c55e;">
                    📡 Sensor Network: AKTIV
                </span>
                <span style="background:rgba(56,189,248,0.2);border:1px solid #38bdf8;
                             padding:4px 12px;border-radius:999px;font-size:12px;color:#38bdf8;">
                    🕐 {current_shift}
                </span>
                <span style="background:rgba(168,85,247,0.2);border:1px solid #a855f7;
                             padding:4px 12px;border-radius:999px;font-size:12px;color:#a855f7;">
                    🔄 Live: {now_str}
                </span>
                <span style="background:rgba(245,158,11,0.2);border:1px solid #f59e0b;
                             padding:4px 12px;border-radius:999px;font-size:12px;color:#f59e0b;">
                    ✅ ISO 9001:2015 Zertifiziert
                </span>
            </div>
        </div>
        <div style="text-align:right;color:#94a3b8;font-size:12px;min-width:200px;">
            <div style="font-size:28px;font-weight:800;color:white;">LTG</div>
            <div>LogisTech GmbH</div>
            <div>Hafenstraße 23</div>
            <div>20457 Hamburg</div>
            <div style="margin-top:6px;color:#64748b;">
                Gegründet {FACTORY_INFO['gegruendet']} | 
                {FACTORY_INFO['mitarbeiter']} Mitarbeiter |
                {FACTORY_INFO['flotte']} Gabelstapler
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Sidebar Controls
# =========================================================
st.sidebar.header("⚙️ Systemkonfiguration")

scenario = st.sidebar.selectbox(
    "Szenario",
    [
        "Normalbetrieb",
        "Wartungskrise",
        "Schichtende",
        "Hochbetrieb"
    ],
    index=0
)

n_stapler = st.sidebar.slider(
    "Anzahl Gabelstapler", 4, 12, 8, 1
)

global_noise = st.sidebar.slider(
    "Sensor-Rauschen", 0.01, 0.15, 0.05, 0.005
)

seed = st.sidebar.number_input(
    "Szenario Seed",
    min_value=1,
    max_value=99999,
    value=123,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Wartungsparameter")

techniker_queue = st.sidebar.slider(
    "Verfügbare Techniker", 1, 6, 2
)

sicherheitsmarge = st.sidebar.slider(
    "Sicherheitsmarge [h]", 0, 10, 3
)

st.sidebar.markdown("---")
st.sidebar.subheader("KI-Entscheidungsschwellen")

auto_threshold = st.sidebar.slider(
    "Auto-Wartungsauftrag ab Confidence",
    0.50, 1.00, 0.82, 0.01
)

manual_threshold = st.sidebar.slider(
    "Technikerfreigabe ab Confidence",
    0.30, 0.90, 0.60, 0.01
)

st.sidebar.caption(
    f"🔄 Live Update alle 5s | {time.strftime('%H:%M:%S')} | #{st.session_state.update_counter}"
)


# =========================================================
# Fleet Evaluation
# =========================================================
with st.spinner("🤖 KI analysiert Gabelstapler-Flotte..."):
    gabelstapler_df = build_gabelstapler_fleet(
        n_stapler=n_stapler,
        scenario=scenario,
        seed=int(seed)
    )
    fleet = evaluate_gabelstapler_fleet(
        fleet_df=gabelstapler_df,
        global_noise=global_noise,
        sicherheitsmarge=sicherheitsmarge,
        techniker_queue=techniker_queue,
        auto_threshold=auto_threshold,
        manual_threshold=manual_threshold,
        seed=int(seed) + 700
    )

# Live Update
rng_live = np.random.default_rng(int(time.time() / 5))
fleet["RUL_min_h"] = fleet["RUL_min_h"].apply(
    lambda x: max(0.1, x + rng_live.normal(0, 0.5))
).round(1)
fleet["Risk_Score"] = fleet["Risk_Score"].apply(
    lambda x: max(0, x + rng_live.normal(0, 1.5))
).round(1)

# =========================================================
# Global KPIs
# =========================================================
urgent_count = fleet[fleet["Entscheidung"].isin([
    "SOFORT_STOPP", "WARTUNGSAUFTRAG", "TEILE_FEHLEN"
])].shape[0]

manual_count = fleet[fleet["Entscheidung"].isin([
    "TECHNIKER_FREIGABE", "UNSICHER_WARNUNG"
])].shape[0]

avg_rul = fleet["RUL_min_h"].mean()
risk_total = fleet["Risk_Score"].sum()

sofort_count = len(fleet[fleet["Entscheidung"] == "SOFORT_STOPP"])
gut_count = len(fleet[fleet["KI_Zustand"] == "Gut"])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    kpi_card(
        "Gabelstapler",
        n_stapler,
        "überwachte Flotte",
        "#38bdf8"
    )
with col2:
    kpi_card(
        "Kritische Wartungen",
        urgent_count,
        "Sofort / Auftrag / Teile",
        "#ef4444"
    )
with col3:
    kpi_card(
        "Techniker Freigaben",
        manual_count,
        "manuelle Entscheidung",
        "#f59e0b"
    )
with col4:
    kpi_card(
        "Ø RUL",
        f"{avg_rul:.1f} h",
        "Restbetriebszeit",
        "#22c55e"
    )
with col5:
    kpi_card(
        "Risiko-Index",
        f"{risk_total:.0f}",
        "aggregierter Wartungsdruck",
        "#a855f7"
    )

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚜 Flotten-Leitwarte",
    "🔍 Stapler-Detail & KI",
    "🔧 Wartung & Planung",
    "📊 KPI & Berichte",
    "🧩 Architektur",
    "🤖 KI-Assistent"
])

# =========================================================
# Tab 1: Flotten-Leitwarte
# =========================================================
with tab1:
    st.header("🚜 Smart Logistics Flotten-Leitwarte")

    # Cost Counter
    elapsed_seconds = time.time() - st.session_state.start_time
    elapsed_minutes = elapsed_seconds / 60
    avg_cost = fleet["Stillstandskosten_EUR_h"].mean()
    total_savings = elapsed_minutes / 60 * avg_cost * 0.42
    avoided_stops = int(elapsed_minutes / 60 / 3)
    avoided_notfall = int(elapsed_minutes / 60 / 2)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#064e3b,#065f46);
                border:1px solid #10b981;border-radius:16px;
                padding:18px;margin-bottom:16px;">
        <div style="font-size:13px;color:#6ee7b7;
                    letter-spacing:2px;margin-bottom:10px;">
            💰 LIVE EINSPARUNGSRECHNER – SEIT SYSTEMSTART
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div>
                <div style="font-size:11px;color:#6ee7b7;">
                    💶 Eingesparte Kosten
                </div>
                <div style="font-size:36px;font-weight:800;color:white;">
                    {total_savings:.2f} €
                </div>
                <div style="font-size:12px;color:#a7f3d0;">
                    seit {int(elapsed_minutes)} Minuten
                </div>
            </div>
            <div>
                <div style="font-size:11px;color:#6ee7b7;">
                    🛑 Vermiedene Ausfälle
                </div>
                <div style="font-size:36px;font-weight:800;color:white;">
                    {avoided_stops}
                </div>
            </div>
            <div>
                <div style="font-size:11px;color:#6ee7b7;">
                    🔧 Vermiedene Notfallwartungen
                </div>
                <div style="font-size:36px;font-weight:800;color:white;">
                    {avoided_notfall}
                </div>
            </div>
            <div>
                <div style="font-size:11px;color:#6ee7b7;">
                    ⏱️ Systemlaufzeit
                </div>
                <div style="font-size:36px;font-weight:800;color:white;">
                    {int(elapsed_minutes)}m {int(elapsed_seconds % 60)}s
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Notifications
    st.subheader("🔔 Live Alarm Center")
    show_notifications(fleet)
    st.markdown("---")

    # Live Feed
    col_live1, col_live2, col_live3, col_live4 = st.columns(4)
    with col_live1:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.1);border:1px solid #22c55e;
                    border-radius:12px;padding:12px;text-align:center;">
            <div style="font-size:11px;color:#64748b;">🏭 WERK</div>
            <div style="font-size:16px;font-weight:700;color:white;">
                {FACTORY_INFO['werk']}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_live2:
        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.1);border:1px solid #38bdf8;
                    border-radius:12px;padding:12px;text-align:center;">
            <div style="font-size:11px;color:#64748b;">🕐 SCHICHT</div>
            <div style="font-size:16px;font-weight:700;color:white;">
                {get_current_shift().split()[0]}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_live3:
        st.markdown(f"""
        <div style="background:rgba(168,85,247,0.1);border:1px solid #a855f7;
                    border-radius:12px;padding:12px;text-align:center;">
            <div style="font-size:11px;color:#64748b;">📡 SENSOREN AKTIV</div>
            <div style="font-size:16px;font-weight:700;color:white;">
                {len(fleet)} / {len(fleet)}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_live4:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.1);border:1px solid #f59e0b;
                    border-radius:12px;padding:12px;text-align:center;">
            <div style="font-size:11px;color:#64748b;">📍 STANDORT</div>
            <div style="font-size:16px;font-weight:700;color:white;">
                Hamburg, DE
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(factory_map(fleet), use_container_width=True)

    st.subheader("Priorisierte Flottenliste")
    display_cols = [
        "Stapler", "Name", "Typ", "Bereich",
        "KI_Zustand", "Confidence", "RUL_min_h",
        "Gesamte_Vorlaufzeit_h", "Entscheidung", "Risk_Score"
    ]
    st.dataframe(
        fleet[display_cols],
        use_container_width=True,
        height=380
    )

    st.subheader("Entscheidungslegende")
    legend_cols = st.columns(4)
    decisions = [
        "MONITORING", "VORWARNUNG", "WARTUNGSAUFTRAG",
        "TECHNIKER_FREIGABE", "UNSICHER_WARNUNG",
        "SOFORT_STOPP", "TEILE_FEHLEN"
    ]
    for i, d in enumerate(decisions):
        with legend_cols[i % 4]:
            st.markdown(badge(d, DECISION_COLORS[d]), unsafe_allow_html=True)

    st.markdown("---")

    # Domino Effect
    st.subheader("🔗 Domino Effekt Analyse")

    domino_stapler = st.selectbox(
        "Simuliere Ausfall von Stapler:",
        fleet["Stapler"].tolist(),
        index=0,
        key="domino_t1"
    )

    domino_row = fleet[fleet["Stapler"] == domino_stapler].iloc[0]
    stapler_info_d = GABELSTAPLER_REGISTRY.get(domino_stapler, {})

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#7f1d1d,#dc2626);
                border-radius:16px;padding:18px;margin:12px 0;">
        <div style="font-size:13px;color:#fca5a5;margin-bottom:4px;">
            ⚠️ SIMULIERTER STAPLERAUSFALL
        </div>
        <div style="font-size:24px;font-weight:800;color:white;">
            {domino_stapler} – {domino_row['Name']}
        </div>
        <div style="color:#fca5a5;font-size:13px;">
            {domino_row['Typ']} | 
            {domino_row['Bereich']} | 
            Zustand: {domino_row['KI_Zustand']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    domino_df = calculate_domino_effect(fleet, domino_stapler)

    if domino_df.empty:
        st.success("✅ Keine anderen Stapler direkt betroffen.")
    else:
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            kpi_card(
                "Betroffene Stapler",
                len(domino_df),
                "direkt beeinflusst",
                "#ef4444"
            )
        with cd2:
            kpi_card(
                "Hohes Risiko",
                len(domino_df[domino_df["Impact_Score"] >= 50]),
                "Impact Score ≥ 50",
                "#f59e0b"
            )
        with cd3:
            kpi_card(
                "Kosten-Risiko",
                f"{domino_df['Stillstandskosten_EUR_h'].sum()} €/h",
                "bei Kettenreaktion",
                "#dc2626"
            )

        col_dl, col_dr = st.columns([1.2, 1])
        with col_dl:
            st.dataframe(
                domino_df[[
                    "Stapler", "Name", "Bereich",
                    "KI_Zustand", "RUL_min_h",
                    "Impact_Score", "Gründe"
                ]],
                use_container_width=True,
                height=280,
                hide_index=True
            )
        with col_dr:
            fig_domino = px.bar(
                domino_df.head(8),
                x="Stapler",
                y="Impact_Score",
                color="KI_Zustand",
                color_discrete_map=STATE_COLORS,
                title="Impact Score",
                text="Impact_Score"
            )
            fig_domino.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                font=dict(color="white"),
                height=280
            )
            st.plotly_chart(fig_domino, use_container_width=True)
# =========================================================
# Tab 2: Stapler Detail & KI
# =========================================================
with tab2:
    st.header("🔍 Stapler-Detail & KI Analyse")

    selected_stapler = st.selectbox(
        "Gabelstapler auswählen",
        fleet["Stapler"].tolist(),
        index=0
    )

    selected = fleet[fleet["Stapler"] == selected_stapler].iloc[0]
    stapler_info = GABELSTAPLER_REGISTRY.get(selected_stapler, {})
    sensor_data = get_sensor_reading(
        stapler_id=selected_stapler,
        zustand=selected["Ist_Zustand"],
        betriebsstunden=selected["Betriebsstunden"],
        seed=int(time.time() / 5)
    )

    # KPI Cards
    colA, colB, colC, colD = st.columns(4)
    with colA:
        kpi_card("Stapler", selected["Stapler"],
                selected["Typ"], "#38bdf8")
    with colB:
        kpi_card("Bereich", selected["Bereich"],
                f"Bediener: {selected['Bediener']}", "#22c55e")
    with colC:
        kpi_card("KI-Zustand", selected["KI_Zustand"],
                f"Confidence {selected['Confidence']*100:.1f}%",
                STATE_COLORS[selected["KI_Zustand"]])
    with colD:
        kpi_card("Entscheidung", selected["Entscheidung"],
                f"Risk Score {selected['Risk_Score']}",
                DECISION_COLORS[selected["Entscheidung"]])

    st.markdown("---")

    # Stapler Identität
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                border:1px solid #334155;border-radius:16px;
                padding:18px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:11px;color:#64748b;
                            letter-spacing:2px;margin-bottom:4px;">
                    STAPLER-IDENTITÄT
                </div>
                <div style="font-size:22px;font-weight:800;color:white;">
                    {selected['Name']}
                </div>
                <div style="color:#94a3b8;font-size:13px;">
                    {selected['Typ']} | 
                    Baujahr {stapler_info.get('baujahr','')} | 
                    Tragkraft: {stapler_info.get('tragkraft_kg','')} kg
                </div>
                <div style="color:#94a3b8;font-size:13px;margin-top:4px;">
                    🔋 {stapler_info.get('batterietyp','')} | 
                    ⏱️ {selected['Betriebsstunden']} Betriebsstunden
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px;color:#64748b;margin-bottom:4px;">
                    SENSOR ID
                </div>
                <div style="font-family:monospace;color:#38bdf8;font-size:14px;">
                    {sensor_data['sensor_id']}
                </div>
                <div style="color:#94a3b8;font-size:12px;margin-top:4px;">
                    Messung #{sensor_data['messung_nr']}
                </div>
                <div style="color:#64748b;font-size:11px;">
                    {sensor_data['timestamp']}
                </div>
            </div>
        </div>

        <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:10px 16px;min-width:120px;">
                <div style="font-size:11px;color:#64748b;">📳 Vibration</div>
                <div style="font-size:20px;font-weight:700;color:#a855f7;">
                    {sensor_data['vibration_mm_s']} mm/s
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:10px 16px;min-width:120px;">
                <div style="font-size:11px;color:#64748b;">🌡️ Motor Temp</div>
                <div style="font-size:20px;font-weight:700;color:#f59e0b;">
                    {sensor_data['motor_temp_c']}°C
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:10px 16px;min-width:120px;">
                <div style="font-size:11px;color:#64748b;">🔋 Batterie</div>
                <div style="font-size:20px;font-weight:700;color:#22c55e;">
                    {sensor_data['batterie_pct']}%
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:10px 16px;min-width:120px;">
                <div style="font-size:11px;color:#64748b;">⚡ Motorstrom</div>
                <div style="font-size:20px;font-weight:700;color:#38bdf8;">
                    {sensor_data['motorstrom_a']} A
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:10px 16px;min-width:120px;">
                <div style="font-size:11px;color:#64748b;">🔧 Hydraulik</div>
                <div style="font-size:20px;font-weight:700;color:#06b6d4;">
                    {sensor_data['hydraulikdruck_bar']} bar
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:10px 16px;min-width:120px;">
                <div style="font-size:11px;color:#64748b;">🔄 Ladezyklen</div>
                <div style="font-size:20px;font-weight:700;color:#f472b6;">
                    {sensor_data['ladezyklen']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge Charts
    st.subheader("📈 Live Sensor Monitor")
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.plotly_chart(
            make_gauge("📳 Vibration",
                      sensor_data["vibration_mm_s"],
                      0, 10, " mm/s", "#a855f7"),
            use_container_width=True
        )
    with g2:
        st.plotly_chart(
            make_gauge("🌡️ Motor Temp",
                      sensor_data["motor_temp_c"],
                      0, 120, "°C", "#f59e0b"),
            use_container_width=True
        )
    with g3:
        st.plotly_chart(
            make_gauge("🔋 Batterie",
                      sensor_data["batterie_pct"],
                      0, 100, "%", "#22c55e"),
            use_container_width=True
        )
    with g4:
        st.plotly_chart(
            make_gauge("⏱️ RUL",
                      selected["RUL_min_h"],
                      0, 120, " h", "#38bdf8"),
            use_container_width=True
        )

    st.markdown("---")

    # Komponenten RUL
    st.subheader("🧬 Komponenten-Gesundheit")

    komp_data = []
    for komp in KOMPONENTEN:
        if komp in selected:
            rul_val = selected[komp]
            grenzen = WARTUNG_GRENZEN[komp]
            if rul_val < grenzen["kritisch"]:
                status = "🚨 Kritisch"
                color = "#ef4444"
            elif rul_val < grenzen["warnung"]:
                status = "⚠️ Warnung"
                color = "#f59e0b"
            else:
                status = "✅ Gut"
                color = "#22c55e"

            komp_data.append({
                "Komponente": komp,
                "RUL [h]": rul_val,
                "Status": status,
                "Kritisch unter [h]": grenzen["kritisch"],
                "Warnung unter [h]": grenzen["warnung"]
            })

    komp_df = pd.DataFrame(komp_data)

    fig_komp = px.bar(
        komp_df,
        x="Komponente",
        y="RUL [h]",
        color="Komponente",
        color_discrete_map=KOMPONENTEN_COLORS,
        title="Verbleibende Betriebszeit pro Komponente",
        text="RUL [h]"
    )

    for komp in KOMPONENTEN:
        grenzen = WARTUNG_GRENZEN[komp]
        fig_komp.add_hline(
            y=grenzen["kritisch"],
            line_dash="dash",
            line_color="#ef4444",
            opacity=0.5
        )

    fig_komp.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=350
    )
    st.plotly_chart(fig_komp, use_container_width=True)
    st.dataframe(komp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Wartungsinfo
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📋 Auftragsinformation")
        st.dataframe(pd.DataFrame({
            "Parameter": [
                "Bereich", "Auftrag", "Batterietyp",
                "Tragkraft", "Betriebsstunden",
                "Fällig in", "Stillstandskosten"
            ],
            "Wert": [
                selected["Bereich"],
                selected["Auftrag"],
                stapler_info.get("batterietyp", "-"),
                f"{stapler_info.get('tragkraft_kg', '-')} kg",
                f"{selected['Betriebsstunden']} h",
                f"{selected['Fällig_in_h']} h",
                f"{selected['Stillstandskosten_EUR_h']} €/h"
            ]
        }), use_container_width=True, hide_index=True)

    with right:
        st.subheader("🔧 Wartungsplanung")
        st.dataframe(pd.DataFrame({
            "Kennzahl": [
                "RUL (minimum)", "Gesamte Vorlaufzeit",
                "Diagnose", "Teile holen",
                "Techniker Warten", "Wartung selbst",
                "Sicherheitsmarge", "Teile Verzögerung",
                "Teile OK"
            ],
            "Wert": [
                f"{selected['RUL_min_h']} h",
                f"{selected['Gesamte_Vorlaufzeit_h']} h",
                f"{selected['Diagnose_h']} h",
                f"{selected['Teile_holen_h']} h",
                f"{selected['Techniker_Warten_h']} h",
                f"{selected['Wartung_h']} h",
                f"{selected['Sicherheitsmarge_h']} h",
                f"{selected['Teile_Verzoegerung_h']} h",
                "Ja" if selected["Teile_OK"] else "Nein"
            ]
        }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Decision Status
    decision = selected["Entscheidung"]
    if decision == "WARTUNGSAUFTRAG":
        st.success("✅ Automatischer Wartungsauftrag wurde erzeugt.")
    elif decision == "SOFORT_STOPP":
        st.error("🚨 Sofortiger Stopp! Stapler sofort zur Werkstatt.")
    elif decision == "TECHNIKER_FREIGABE":
        st.warning("⚠️ Technikerfreigabe erforderlich.")
    elif decision == "TEILE_FEHLEN":
        st.error("❌ Ersatzteile nicht verfügbar! Sonderbeschaffung nötig.")
    elif decision == "VORWARNUNG":
        st.info("🔔 Vorwarnung: Wartung bald erforderlich.")
    else:
        st.info("✅ Monitoring: Stapler läuft normal.")

# =========================================================
# Tab 3: Wartung & Planung
# =========================================================
with tab3:
    st.header("🔧 Wartung & Planung")

    # Order Board
    order_filter = fleet[fleet["Entscheidung"].isin([
        "SOFORT_STOPP", "TEILE_FEHLEN", "WARTUNGSAUFTRAG",
        "TECHNIKER_FREIGABE", "UNSICHER_WARNUNG", "VORWARNUNG"
    ])].copy()

    if order_filter.empty:
        st.info("Aktuell keine Wartungsaufträge notwendig.")
    else:
        st.subheader("📋 Wartungs-Order Board")
        st.dataframe(
            order_filter[[
                "Stapler", "Name", "Bereich", "KI_Zustand",
                "RUL_min_h", "Gesamte_Vorlaufzeit_h",
                "Entscheidung", "Teile_OK", "Risk_Score"
            ]],
            use_container_width=True,
            height=300
        )

    st.markdown("---")

    colL, colR = st.columns(2)

    with colL:
        st.subheader("👨‍🔧 Techniker Status")
        tech_rows = []
        for i in range(1, techniker_queue + 2):
            if i <= len(order_filter):
                r = order_filter.iloc[i-1]
                tech_rows.append({
                    "Techniker": f"Tech-{i:02d}",
                    "Status": f"Wartung {r['Stapler']}",
                    "Bereich": r["Bereich"],
                    "Restzeit": f"{r['Wartung_h']} h"
                })
            else:
                tech_rows.append({
                    "Techniker": f"Tech-{i:02d}",
                    "Status": "Verfügbar",
                    "Bereich": "-",
                    "Restzeit": "-"
                })
        st.dataframe(
            pd.DataFrame(tech_rows),
            use_container_width=True,
            hide_index=True
        )

    with colR:
        st.subheader("🏪 Ersatzteillager Status")
        teile_rows = []
        komponenten_liste = [
            ("Batterie 48V Li-Ion", "BT-48V-LI", 3),
            ("Motor Antriebseinheit", "MOT-AE-24", 2),
            ("Vollgummi-Reifen Vorne", "RE-VG-F", 8),
            ("Hydraulikpumpe", "HY-PMP-01", 1),
            ("Bremsenkit komplett", "BR-KIT-01", 4),
        ]
        for name, art_nr, bestand in komponenten_liste:
            status = "✅ OK" if bestand > 2 else ("⚠️ Niedrig" if bestand > 0 else "🚨 Leer")
            teile_rows.append({
                "Ersatzteil": name,
                "Art-Nr": art_nr,
                "Bestand": bestand,
                "Status": status
            })
        st.dataframe(
            pd.DataFrame(teile_rows),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Wartungskalender
    st.subheader("📅 Predictiver Wartungskalender")

    now_cal = pd.Timestamp.now()
    calendar_data = []

    for _, row in fleet.iterrows():
        s_info = GABELSTAPLER_REGISTRY.get(row["Stapler"], {})
        wartung_time = now_cal + pd.Timedelta(hours=float(row["RUL_min_h"]))

        if row["RUL_min_h"] < 10:
            priority, color = "🚨 Kritisch", "#ef4444"
        elif row["RUL_min_h"] < 24:
            priority, color = "⚠️ Dringend", "#f59e0b"
        elif row["RUL_min_h"] < 48:
            priority, color = "🔔 Bald", "#6366f1"
        else:
            priority, color = "✅ Normal", "#22c55e"

        calendar_data.append({
            "Stapler": row["Stapler"],
            "Name": row["Name"],
            "Bereich": row["Bereich"],
            "RUL_h": row["RUL_min_h"],
            "Wartung_um": wartung_time.strftime("%d.%m.%Y %H:%M"),
            "Priorität": priority,
            "Farbe": color,
            "Bediener": s_info.get("bediener", "-")
        })

    cal_df = pd.DataFrame(calendar_data).sort_values(
        "RUL_h", ascending=True
    ).reset_index(drop=True)

    ck1, ck2, ck3, ck4 = st.columns(4)
    with ck1:
        kpi_card("🚨 Kritisch",
                len(cal_df[cal_df["RUL_h"] < 10]),
                "< 10 Stunden", "#ef4444")
    with ck2:
        kpi_card("⚠️ Dringend",
                len(cal_df[(cal_df["RUL_h"] >= 10) &
                           (cal_df["RUL_h"] < 24)]),
                "10-24 Stunden", "#f59e0b")
    with ck3:
        kpi_card("🔔 Bald",
                len(cal_df[(cal_df["RUL_h"] >= 24) &
                           (cal_df["RUL_h"] < 48)]),
                "24-48 Stunden", "#6366f1")
    with ck4:
        kpi_card("✅ Normal",
                len(cal_df[cal_df["RUL_h"] >= 48]),
                "> 48 Stunden", "#22c55e")

    fig_cal = go.Figure()
    for _, row in cal_df.iterrows():
        fig_cal.add_trace(go.Bar(
            x=[row["RUL_h"]],
            y=[f"{row['Stapler']} – {row['Name']}"],
            orientation="h",
            marker=dict(color=row["Farbe"]),
            text=f"{row['RUL_h']} h | {row['Wartung_um']}",
            textposition="inside",
            showlegend=False
        ))

    fig_cal.add_vline(
        x=10, line_dash="dash", line_color="#ef4444",
        annotation_text="🚨 Kritisch",
        annotation_font_color="#ef4444"
    )
    fig_cal.add_vline(
        x=24, line_dash="dash", line_color="#f59e0b",
        annotation_text="⚠️ Dringend",
        annotation_font_color="#f59e0b"
    )

    fig_cal.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=450,
        xaxis=dict(
            title="Verbleibende Zeit [h]",
            gridcolor="#334155",
            color="white"
        ),
        yaxis=dict(gridcolor="#334155", color="white"),
        title=dict(
            text="Wartungsplan – Alle Stapler nach RUL sortiert",
            font=dict(color="white")
        )
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.dataframe(
        cal_df[[
            "Priorität", "Stapler", "Name", "Bereich",
            "RUL_h", "Wartung_um", "Bediener"
        ]],
        use_container_width=True,
        hide_index=True
    )

    csv_data = cal_df.to_csv(index=False)
    st.download_button(
        label="📥 Wartungsplan herunterladen",
        data=csv_data,
        file_name=f"Wartungsplan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Energy Optimizer
    st.subheader("⚡ Energy Optimizer – Batterieladung")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a8a,#1e293b);
                border:1px solid #3b82f6;border-radius:12px;
                padding:14px;margin-bottom:16px;">
        <div style="color:#93c5fd;font-size:13px;">
            ⚡ Das System berechnet den optimalen Ladezeitpunkt für jeden
            Elektrostapler basierend auf Schichtplan, Batteriestatus und
            geplantem Einsatz – um Produktionsunterbrechungen zu vermeiden.
        </div>
    </div>
    """, unsafe_allow_html=True)

    elektro_fleet = fleet[
        fleet["Batterietyp"].str.contains("Li-Ion|Blei", na=False)
    ] if "Batterietyp" in fleet.columns else fleet

    energy_data = []
    rng_e = np.random.default_rng(int(seed) + 300)

    for _, row in fleet.iterrows():
        s_info = GABELSTAPLER_REGISTRY.get(row["Stapler"], {})
        bat_typ = s_info.get("batterietyp", "")

        if "Li-Ion" in bat_typ or "Blei" in bat_typ:
            batt_pct = get_sensor_reading(
                row["Stapler"],
                row["Ist_Zustand"],
                row["Betriebsstunden"],
                seed=int(seed) + hash(row["Stapler"]) % 100
            )["batterie_pct"]

            restzeit_h = round(batt_pct / 100 * rng_e.uniform(6, 10), 1)
            lade_empfehlung = "Jetzt laden" if batt_pct < 30 else (
                "Bald laden" if batt_pct < 50 else "OK"
            )
            lade_color = "#ef4444" if batt_pct < 30 else (
                "#f59e0b" if batt_pct < 50 else "#22c55e"
            )

            energy_data.append({
                "Stapler": row["Stapler"],
                "Name": row["Name"],
                "Batterietyp": bat_typ,
                "🔋 Ladung %": round(batt_pct, 1),
                "⏱️ Restzeit [h]": restzeit_h,
                "💡 Empfehlung": lade_empfehlung,
                "Farbe": lade_color
            })

    if energy_data:
        energy_df = pd.DataFrame(energy_data)

        fig_energy = px.bar(
            energy_df,
            x="Stapler",
            y="🔋 Ladung %",
            color="💡 Empfehlung",
            color_discrete_map={
                "Jetzt laden": "#ef4444",
                "Bald laden": "#f59e0b",
                "OK": "#22c55e"
            },
            title="Batteriestatus – Alle Elektrostapler",
            text="🔋 Ladung %"
        )

        fig_energy.add_hline(
            y=30, line_dash="dash", line_color="#ef4444",
            annotation_text="⚠️ Sofort laden",
            annotation_font_color="#ef4444"
        )
        fig_energy.add_hline(
            y=50, line_dash="dash", line_color="#f59e0b",
            annotation_text="Bald laden",
            annotation_font_color="#f59e0b"
        )

        fig_energy.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            font=dict(color="white"),
            height=380
        )
        st.plotly_chart(fig_energy, use_container_width=True)

        st.dataframe(
            energy_df[[
                "Stapler", "Name", "Batterietyp",
                "🔋 Ladung %", "⏱️ Restzeit [h]", "💡 Empfehlung"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Keine Elektrostapler in der aktuellen Flotte.")
# =========================================================
# Tab 4: KPI & Berichte
# =========================================================
with tab4:
    st.header("📊 KPI & Berichte")

    st.subheader("📊 KPI Simulation: Traditionell vs. Predictive")

    kpi_events = st.slider(
        "Anzahl simulierter Wartungsereignisse",
        50, 500, 180, 10
    )

    kpi_df, summary = simulate_kpis(
        n_events=kpi_events,
        seed=int(seed) + 400
    )

    traditional = summary[
        summary["Methode"] == "Traditionell"
    ].iloc[0]
    predictive = summary[
        summary["Methode"] == "Predictive Maintenance"
    ].iloc[0]

    downtime_reduction = (
        1 - predictive["Gesamtstillstand_h"] /
        traditional["Gesamtstillstand_h"]
    ) * 100

    notfall_reduction = (
        1 - predictive["Notfallwartungen"] /
        max(traditional["Notfallwartungen"], 1)
    ) * 100

    cost_reduction = (
        1 - predictive["Stillstandskosten_EUR"] /
        traditional["Stillstandskosten_EUR"]
    ) * 100

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        kpi_card(
            "Stillstandsreduktion",
            f"{downtime_reduction:.1f}%",
            "gegenüber traditionell",
            "#22c55e"
        )
    with kc2:
        kpi_card(
            "Notfallwartung-Reduktion",
            f"{notfall_reduction:.1f}%",
            "weniger Notfälle",
            "#38bdf8"
        )
    with kc3:
        kpi_card(
            "Kostenreduktion",
            f"{cost_reduction:.1f}%",
            "Stillstandskosten",
            "#a855f7"
        )

    st.markdown("---")

    ka, kb = st.columns(2)
    with ka:
        fig1 = px.bar(
            summary,
            x="Methode",
            y="Gesamtstillstand_h",
            color="Methode",
            title="Gesamter Stillstand [h]"
        )
        fig1.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            font=dict(color="white")
        )
        st.plotly_chart(fig1, use_container_width=True)

    with kb:
        fig2 = px.bar(
            summary,
            x="Methode",
            y="Notfallwartungen",
            color="Methode",
            title="Anzahl Notfallwartungen"
        )
        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            font=dict(color="white")
        )
        st.plotly_chart(fig2, use_container_width=True)

    kc, kd = st.columns(2)
    with kc:
        fig3 = px.bar(
            summary,
            x="Methode",
            y="Rechtzeitige_Wartung",
            color="Methode",
            title="Rechtzeitige Wartungen [%]"
        )
        fig3.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            font=dict(color="white")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with kd:
        fig4 = px.bar(
            summary,
            x="Methode",
            y="Staplernutzung",
            color="Methode",
            title="Staplernutzung [%]"
        )
        fig4.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            font=dict(color="white")
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Schichtbericht
    st.subheader("📄 Automatischer Schichtbericht")

    now_b = pd.Timestamp.now()
    elapsed_min = (time.time() - st.session_state.start_time) / 60
    total_sav = elapsed_min / 60 * fleet["Stillstandskosten_EUR_h"].mean() * 0.42

    sofort_c = len(fleet[fleet["Entscheidung"] == "SOFORT_STOPP"])
    wartung_c = len(fleet[fleet["Entscheidung"] == "WARTUNGSAUFTRAG"])
    teile_c = len(fleet[fleet["Entscheidung"] == "TEILE_FEHLEN"])
    monitor_c = len(fleet[fleet["Entscheidung"] == "MONITORING"])

    sb1, sb2, sb3 = st.columns(3)
    with sb1:
        kpi_card(
            "Aktuelle Schicht",
            get_current_shift().split()[0],
            get_current_shift(),
            "#38bdf8"
        )
    with sb2:
        kpi_card(
            "Datum",
            now_b.strftime("%d.%m.%Y"),
            now_b.strftime("%H:%M Uhr"),
            "#22c55e"
        )
    with sb3:
        kpi_card(
            "Eingesparte Kosten",
            f"{total_sav:.2f} €",
            "seit Systemstart",
            "#a855f7"
        )

    kritische_b = fleet[fleet["Entscheidung"].isin([
        "SOFORT_STOPP", "WARTUNGSAUFTRAG", "TEILE_FEHLEN"
    ])]

    kritische_liste = ""
    for _, row in kritische_b.iterrows():
        kritische_liste += (
            f"• {row['Stapler']} ({row['Name']}): "
            f"{row['Entscheidung']} – RUL: {row['RUL_min_h']} h\n"
        )

    bericht_text = f"""
SCHICHTBERICHT – LogisTech GmbH – Werk 1, Hamburg
{"="*55}
Datum:         {now_b.strftime("%d.%m.%Y")}
Uhrzeit:       {now_b.strftime("%H:%M:%S")} Uhr
Schicht:       {get_current_shift()}
System:        Predictive Gabelstapler Maintenance (KI)
{"="*55}

ZUSAMMENFASSUNG
Überwachte Stapler:    {len(fleet)}
Sofort-Stopps:         {sofort_c}
Wartungsaufträge:      {wartung_c}
Teile fehlen:          {teile_c}
Normalbetrieb:         {monitor_c}

WIRTSCHAFTLICHE KENNZAHLEN
Systembetrieb:         {int(elapsed_min)} Minuten
Eingesparte Kosten:    {total_sav:.2f} €
Verm. Ausfälle:        {int(elapsed_min/60/3)}
Verm. Notfallwartung:  {int(elapsed_min/60/2)}

KRITISCHE STAPLER
{kritische_liste if kritische_liste else "Keine kritischen Stapler."}

EMPFEHLUNGEN
{"• Sofortige Überprüfung erforderlich!" if sofort_c > 0 else "• Normaler Betrieb kann fortgesetzt werden"}
{"• Ersatzteile für " + str(teile_c) + " Stapler prüfen" if teile_c > 0 else ""}
{"• " + str(wartung_c) + " Wartungsaufträge automatisch erstellt" if wartung_c > 0 else ""}

{"="*55}
SYSTEM: Predictive Gabelstapler Maintenance
KI: Vibration + Sensor Fusion + Gemini AI
{"="*55}
    """

    st.code(bericht_text, language="text")

    st.download_button(
        label="📥 Schichtbericht herunterladen",
        data=bericht_text,
        file_name=f"Schichtbericht_{now_b.strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )
# =========================================================
# Tab 5: Architektur
# =========================================================
with tab5:
    st.header("🧩 Architektur & Innovationskern")

    st.subheader("Systemarchitektur")
    st.code("""
Gabelstapler Vibrations- & Sensordaten
        ↓
Multi-Sensor Fusion
(Vibration + Temperatur + Batterie + Hydraulik + Strom)
        ↓
KI-Modell: Zustandsklassifikation
(Gut / Warnung / Kritisch / Ausfall)
        ↓
RUL-Prognose pro Komponente
(Batterie / Motor / Reifen / Hydraulik / Bremsen)
        ↓
Vergleich mit Wartungsvorlaufzeit
        ↓
Automatische Entscheidung:
    MONITORING / VORWARNUNG / WARTUNGSAUFTRAG
    TECHNIKER_FREIGABE / SOFORT_STOPP / TEILE_FEHLEN
        ↓
Werkstatt → Techniker → Gabelstapler
        ↓
Energy Optimizer: Optimale Ladeplanung
    """, language="text")

    st.subheader("🚀 Innovationskern")
    st.success("""
    Das System ist weltweit einzigartig:
    Es kombiniert Multi-Sensor-Fusion mit KI-basierter
    Komponentendiagnose und automatisierter Wartungslogistik
    für Gabelstapler. Kein anderes System auf dem Markt
    verbindet RUL-Prognose pro Komponente mit automatischer
    Wartungsplanung und Energy Optimizer in Echtzeit.
    """)

    st.subheader("Was macht unser System einzigartig?")
    unique = pd.DataFrame({
        "Feature": [
            "🧬 Komponenten-DNA",
            "⚡ Energy Optimizer",
            "🔗 Domino Effekt",
            "📅 Predictiver Kalender",
            "🤖 Gemini AI Chat",
            "🔔 Live Alarm Center",
            "💰 Live Einsparungsrechner"
        ],
        "Beschreibung": [
            "RUL pro Komponente: Batterie, Motor, Reifen, Hydraulik, Bremsen",
            "Optimaler Ladezeitpunkt für jeden Elektrostapler",
            "Kettenreaktion bei Staplerausfall vorausberechnen",
            "Wartungstermine automatisch basierend auf RUL planen",
            "KI-Assistent kennt alle Live-Daten der Flotte",
            "Echtzeit-Alarme bei kritischen Zuständen",
            "Live-Berechnung der eingesparten Kosten"
        ],
        "Status": [
            "✅ Aktiv",
            "✅ Aktiv",
            "✅ Aktiv",
            "✅ Aktiv",
            "✅ Aktiv",
            "✅ Aktiv",
            "✅ Aktiv"
        ]
    })
    st.dataframe(unique, use_container_width=True, hide_index=True)

    st.subheader("Industrie 4.0 Bausteine")
    ind40 = pd.DataFrame({
        "Baustein": [
            "IoT / Sensorik",
            "KI / Machine Learning",
            "Predictive Analytics",
            "Wartungslogistik",
            "Automatisierung",
            "Digital Twin",
            "Smart Factory",
            "Generative KI"
        ],
        "Umsetzung": [
            "Vibration, Temperatur, Batterie, Hydraulik, Strom",
            "KI-Zustandsklassifikation mit Confidence Score",
            "RUL-Prognose pro Komponente + Energy Optimizer",
            "Techniker, Ersatzteillager, Wartungsplanung",
            "Automatische Wartungsaufträge und Entscheidungen",
            "Digitale Lagerkarte mit Live-Positionen",
            "Control Tower mit Echtzeit-Alarmen",
            "Gemini AI Chat mit Live-Flottendaten"
        ]
    })
    st.dataframe(ind40, use_container_width=True, hide_index=True)

    st.subheader("Technologie-Stack")
    tech = pd.DataFrame({
        "Technologie": [
            "Python + Streamlit",
            "NumPy + Pandas",
            "Plotly",
            "Scikit-Learn",
            "Google Gemini AI",
            "Requests"
        ],
        "Verwendung": [
            "Dashboard und Web-Interface",
            "Datenverarbeitung und Simulation",
            "Interaktive Visualisierungen",
            "KI-Klassifikation",
            "KI-Chat Assistant",
            "API-Kommunikation"
        ]
    })
    st.dataframe(tech, use_container_width=True, hide_index=True)

    st.subheader("Risiken und Gegenmaßnahmen")
    risks = pd.DataFrame({
        "Risiko": [
            "Sensorrauschen verfälscht Diagnose",
            "KI braucht echte Trainingsdaten",
            "Ersatzteile nicht verfügbar",
            "Integration in bestehende Systeme"
        ],
        "Gegenmaßnahme": [
            "Multi-Sensor Fusion + Confidence Score",
            "Pilotphase mit echten Gabelstapler-Daten",
            "Automatische Bestandsprüfung + Bestellung",
            "REST API + MQTT + OPC-UA Schnittstellen"
        ]
    })
    st.dataframe(risks, use_container_width=True, hide_index=True)

    st.subheader("Präsentations-Pitch")
    st.info("""
    Unser System „Predictive Gabelstapler Maintenance" ist eine
    weltweit einzigartige Lösung für die vorausschauende Wartung
    von Gabelstaplern in Logistikzentren.

    Das System überwacht kontinuierlich Vibration, Temperatur,
    Batteriestatus, Hydraulikdruck und Motorstrom jedes Staplers.
    Eine KI klassifiziert den Zustand und prognostiziert die
    verbleibende Betriebszeit jeder Komponente einzeln.

    Wenn die Zeit nicht mehr ausreicht, wird automatisch ein
    Wartungsauftrag erstellt, Techniker werden informiert und
    Ersatzteile werden reserviert – alles bevor der Stapler ausfällt.

    Zusätzlich optimiert das System die Ladezeiten der
    Elektrostapler, um Produktionsunterbrechungen zu minimieren.
    """)

# =========================================================
# Tab 6: KI-Assistent
# =========================================================
with tab6:
    st.header("🤖 KI-Assistent – Powered by Gemini AI")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1b4b,#312e81);
                border:1px solid #6366f1;border-radius:12px;
                padding:12px;margin-bottom:16px;">
        <div style="color:#a5b4fc;font-size:13px;">
            🧠 Dieser Assistent kennt alle Live-Daten der Gabelstapler-Flotte
            von LogisTech GmbH. Fragen auf Deutsch, Englisch oder Arabisch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        GEMINI_API_KEY = st.secrets["AQ.Ab8RN6LuCcnYtOWaDmrUseE13ppnI-Dh8q7FnlJLi9U31jpTqw"]
    except:
        GEMINI_API_KEY = ""
        st.warning("""
        ⚠️ Kein API Key gefunden.
        Bitte in Streamlit Secrets hinzufügen: GEMINI_API_KEY
        """)

    def build_fleet_context():
        kritisch = fleet[fleet["Entscheidung"].isin([
            "SOFORT_STOPP", "WARTUNGSAUFTRAG", "TEILE_FEHLEN"
        ])]

        return f"""
Du bist der KI-Assistent des Predictive Gabelstapler
Maintenance Systems der LogisTech GmbH – Werk 1, Hamburg.

AKTUELLER FLOTTENSTATUS:
- Gesamtstapler: {len(fleet)}
- Kritische Stapler: {len(kritisch)}
- Durchschnittliche RUL: {fleet['RUL_min_h'].mean():.1f} h
- Risiko-Index: {fleet['Risk_Score'].sum():.0f}
- Aktuelle Schicht: {get_current_shift()}

KRITISCHE STAPLER:
{kritisch[['Stapler', 'Name', 'KI_Zustand', 'RUL_min_h', 'Entscheidung']].to_string() if len(kritisch) > 0 else 'Keine kritischen Stapler'}

ALLE STAPLER:
{fleet[['Stapler', 'Name', 'KI_Zustand', 'RUL_min_h', 'Entscheidung', 'Risk_Score']].to_string()}

Beantworte Fragen auf Deutsch, Englisch oder Arabisch.
Sei präzise, professionell und hilfreich.
        """

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    frage = st.chat_input(
        "Frage auf Deutsch, Englisch oder Arabisch...",
        key="gemini_chat_final"
    )

    if frage:
        st.session_state.chat_history.append({
            "role": "user",
            "content": frage
        })

        with st.chat_message("user"):
            st.markdown(frage)

        with st.chat_message("assistant"):
            with st.spinner("KI analysiert Flottendaten..."):
                try:
                    url = (
                        f"https://generativelanguage.googleapis.com"
                        f"/v1beta/models/gemini-2.5-flash"
                        f":generateContent?key={AQ.Ab8RN6LuCcnYtOWaDmrUseE13ppnI-Dh8q7FnlJLi9U31jpTqw}"
                    )

                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": (
                                            f"{build_fleet_context()}"
                                            f"\n\nFrage: {frage}"
                                        )
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 1000
                        }
                    }

                    response = requests.post(
                        url,
                        json=payload,
                        timeout=15
                    )

                    if response.status_code == 200:
                        data = response.json()
                        try:
                            antwort = data["candidates"][0]["content"]["parts"][0]["text"]
                            if not antwort or antwort.strip() == "":
                                antwort = "⚠️ Keine Antwort. Bitte erneut versuchen."
                        except (KeyError, IndexError) as e:
                            antwort = f"⚠️ Fehler: {str(e)}\nResponse: {str(data)}"
                    else:
                        antwort = (
                            f"⚠️ API Fehler: {response.status_code}\n"
                            f"{response.text}"
                        )

                except Exception as e:
                    antwort = f"⚠️ Verbindungsfehler: {str(e)}"

                st.markdown(antwort)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": antwort
        })

    col_chat, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Löschen", key="clear_final"):
            st.session_state.chat_history = []
            st.rerun()
