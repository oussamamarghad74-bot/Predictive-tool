import streamlit as st
import pandas as pd
import plotly.express as px


import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# Global Settings
# =========================================================

SR = 12000
DURATION = 2.5

CLASS_ORDER = ["Healthy", "Warning", "Critical", "Replace"]

RUL_BASE = {
    "Healthy": 95,
    "Warning": 58,
    "Critical": 30,
    "Replace": 7
}

SEVERITY = {
    "Healthy": 0,
    "Warning": 1,
    "Critical": 2,
    "Replace": 3
}

STATE_COLORS = {
    "Healthy": "#16a34a",
    "Warning": "#f59e0b",
    "Critical": "#ef4444",
    "Replace": "#7f1d1d"
}

DECISION_COLORS = {
    "MONITORING": "#2563eb",
    "VORWARNUNG": "#7c3aed",
    "AUTO_AUFTRAG": "#16a34a",
    "BEDIENER_FREIGABE": "#f59e0b",
    "UNSICHER_WARNUNG": "#fb7185",
    "SOFORT_STOPP": "#dc2626",
    "BESTANDSRISIKO": "#b45309"
}

# =========================================================
# FertigungsTech GmbH – Virtuelle Fabrik
# =========================================================

FACTORY_INFO = {
    "name": "FertigungsTech GmbH",
    "werk": "Werk 1 – München",
    "adresse": "Industriestraße 47, 80339 München",
    "schichten": ["Frühschicht 06:00–14:00", "Spätschicht 14:00–22:00", "Nachtschicht 22:00–06:00"],
    "gegruendet": "2008",
    "mitarbeiter": 347,
    "zertifikate": ["ISO 9001:2015", "DIN EN 13849", "VDI 2853"]
}

MACHINE_REGISTRY = {
    "M01": {"name": "DMG MORI DMU 50",     "typ": "5-Achs-Fräszentrum",     "zelle": "Zelle A", "bediener": "K. Weber",   "sensor_id": "SNS-2024-M01-ACC", "baujahr": 2019},
    "M02": {"name": "HAAS VF-4",           "typ": "3-Achs-Fräszentrum",     "zelle": "Zelle A", "bediener": "T. Müller",  "sensor_id": "SNS-2024-M02-ACC", "baujahr": 2020},
    "M03": {"name": "Mazak VARIAXIS i-700","typ": "5-Achs-Bearbeitungszentrum","zelle": "Zelle A","bediener": "F. Schmidt", "sensor_id": "SNS-2024-M03-ACC", "baujahr": 2021},
    "M04": {"name": "DMG MORI NLX 2500",   "typ": "CNC-Drehmaschine",       "zelle": "Zelle A", "bediener": "R. Bauer",   "sensor_id": "SNS-2024-M04-ACC", "baujahr": 2018},
    "M05": {"name": "Trumpf TruLaser 3030","typ": "Laserschneidanlage",      "zelle": "Zelle A", "bediener": "M. Klein",   "sensor_id": "SNS-2024-M05-ACC", "baujahr": 2022},
    "M06": {"name": "HAAS ST-30",          "typ": "CNC-Drehmaschine",       "zelle": "Zelle A", "bediener": "S. Hoffmann","sensor_id": "SNS-2024-M06-ACC", "baujahr": 2020},
    "M07": {"name": "Hermle C 400",        "typ": "5-Achs-Fräszentrum",     "zelle": "Zelle B", "bediener": "A. Fischer", "sensor_id": "SNS-2024-M07-ACC", "baujahr": 2021},
    "M08": {"name": "Mazak INTEGREX i-400","typ": "Dreh-Fräszentrum",       "zelle": "Zelle B", "bediener": "P. Wagner",  "sensor_id": "SNS-2024-M08-ACC", "baujahr": 2019},
    "M09": {"name": "DMG MORI CTX 450",    "typ": "CNC-Drehmaschine",       "zelle": "Zelle B", "bediener": "L. Braun",   "sensor_id": "SNS-2024-M09-ACC", "baujahr": 2020},
    "M10": {"name": "HAAS VF-6",           "typ": "3-Achs-Fräszentrum",     "zelle": "Zelle B", "bediener": "J. Schulz",  "sensor_id": "SNS-2024-M10-ACC", "baujahr": 2022},
    "M11": {"name": "Okuma GENOS M460V",   "typ": "Bearbeitungszentrum",    "zelle": "Zelle B", "bediener": "C. Richter", "sensor_id": "SNS-2024-M11-ACC", "baujahr": 2021},
    "M12": {"name": "Mazak QT-PRIMOS 150", "typ": "CNC-Drehmaschine",       "zelle": "Zelle B", "bediener": "H. Wolf",    "sensor_id": "SNS-2024-M12-ACC", "baujahr": 2018},
    "M13": {"name": "DMG MORI DMU 65",     "typ": "5-Achs-Fräszentrum",     "zelle": "Zelle C", "bediener": "G. Neumann", "sensor_id": "SNS-2024-M13-ACC", "baujahr": 2022},
    "M14": {"name": "FANUC Robodrill",     "typ": "Bohrzentrum",            "zelle": "Zelle C", "bediener": "D. Schwarz", "sensor_id": "SNS-2024-M14-ACC", "baujahr": 2020},
    "M15": {"name": "Heller MCH 250",      "typ": "Horizontal-Bearbeitungszentrum","zelle": "Zelle C","bediener": "E. Zimmermann","sensor_id": "SNS-2024-M15-ACC", "baujahr": 2019},
    "M16": {"name": "DMG MORI NHX 4000",   "typ": "Horizontal-Fräszentrum", "zelle": "Zelle C", "bediener": "B. Krause",  "sensor_id": "SNS-2024-M16-ACC", "baujahr": 2021},
    "M17": {"name": "Waldrich Coburg",     "typ": "Schwerzerspanung",       "zelle": "Zelle D", "bediener": "N. Hartmann","sensor_id": "SNS-2024-M17-ACC", "baujahr": 2017},
    "M18": {"name": "TOS WHQ 13",          "typ": "Waagerecht-Bohrwerk",    "zelle": "Zelle D", "bediener": "O. Lange",   "sensor_id": "SNS-2024-M18-ACC", "baujahr": 2016},
    "M19": {"name": "Skoda W 200",         "typ": "Schwerzerspanung",       "zelle": "Zelle D", "bediener": "V. Köhler",  "sensor_id": "SNS-2024-M19-ACC", "baujahr": 2018},
    "M20": {"name": "Forest-Liné Axia",    "typ": "Portalfräsmaschine",     "zelle": "Zelle D", "bediener": "I. Maier",   "sensor_id": "SNS-2024-M20-ACC", "baujahr": 2020},
    "M21": {"name": "DMG MORI DMU 210P",   "typ": "Portal-Fräszentrum",     "zelle": "Zelle D", "bediener": "U. Fuchs",   "sensor_id": "SNS-2024-M21-ACC", "baujahr": 2021},
    "M22": {"name": "Starrag STC 1250",    "typ": "5-Achs-Fräszentrum",     "zelle": "Zelle D", "bediener": "Q. Peters",  "sensor_id": "SNS-2024-M22-ACC", "baujahr": 2022},
    "M23": {"name": "HAAS VF-12",          "typ": "3-Achs-Fräszentrum",     "zelle": "Zelle D", "bediener": "X. Frank",   "sensor_id": "SNS-2024-M23-ACC", "baujahr": 2019},
    "M24": {"name": "Mazak HCN-6800",      "typ": "Horizontal-Bearbeitungszentrum","zelle": "Zelle D","bediener": "Y. Berg","sensor_id": "SNS-2024-M24-ACC", "baujahr": 2020},
}

def get_current_shift():
    hour = pd.Timestamp.now().hour
    if 6 <= hour < 14:
        return "Frühschicht 06:00–14:00"
    elif 14 <= hour < 22:
        return "Spätschicht 14:00–22:00"
    else:
        return "Nachtschicht 22:00–06:00"

def get_sensor_reading(machine_id, state, rpm, seed=0):
    rng = np.random.default_rng(seed)
    
    base_temp = {"Healthy": 52, "Warning": 67, "Critical": 81, "Replace": 94}
    base_vib  = {"Healthy": 0.42, "Warning": 0.78, "Critical": 1.24, "Replace": 1.87}
    base_amp  = {"Healthy": 10.2, "Warning": 12.8, "Critical": 15.4, "Replace": 18.9}
    
    return {
        "sensor_id": MACHINE_REGISTRY.get(machine_id, {}).get("sensor_id", "SNS-UNKNOWN"),
        "timestamp": pd.Timestamp.now().strftime("%d.%m.%Y – %H:%M:%S"),
        "messung_nr": int(rng.integers(4000, 9999)),
        "temperatur_c": round(base_temp[state] + rng.normal(0, 2.5), 1),
        "vibration_mm_s": round(base_vib[state] + rng.normal(0, 0.08), 2),
        "spindelstrom_a": round(base_amp[state] + rng.normal(0, 0.6), 1),
        "drehzahl_rpm": rpm,
        "kuehlmittel_temp_c": round(28.5 + rng.normal(0, 1.2), 1),
        "audio_rms": round(0.03 + SEVERITY[state] * 0.025 + rng.uniform(0, 0.01), 3)
    }
# =========================================================
# Digital Twin Soundtrack - Acoustic Fingerprint
# =========================================================

MACHINE_FINGERPRINTS = {}

def learn_acoustic_fingerprint(machine_id, y, sr=SR):
    """
    تعلم البصمة الصوتية الأساسية لكل آلة
    عندما تكون في حالة Healthy
    """
    features = extract_features(y)
    MACHINE_FINGERPRINTS[machine_id] = {
        "baseline": features,
        "timestamp": pd.Timestamp.now(),
        "learned": True
    }
    return features

def compare_to_fingerprint(machine_id, current_features):
    """
    قارن الصوت الحالي مع البصمة الأساسية
    وأعط نسبة الانحراف
    """
    if machine_id not in MACHINE_FINGERPRINTS:
        return None, "Keine Baseline"
    
    baseline = MACHINE_FINGERPRINTS[machine_id]["baseline"]
    
    # حساب الانحراف بين الصوت الحالي والبصمة
    deviation = np.linalg.norm(current_features - baseline)
    deviation_pct = min(deviation / (np.linalg.norm(baseline) + 1e-9) * 100, 100)
    
    if deviation_pct < 15:
        status = "✅ Normal"
    elif deviation_pct < 35:
        status = "⚠️ Leichte Abweichung"
    elif deviation_pct < 60:
        status = "🔶 Starke Abweichung"
    else:
        status = "🚨 Kritische Abweichung"
    
    return round(deviation_pct, 1), status
# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Predictive Tool Logistics – Advanced Prototype",
    page_icon="🏭",
    layout="wide"
)
import time

if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

if "update_counter" not in st.session_state:
    st.session_state.update_counter = 0

if time.time() - st.session_state.last_update > 5:
    st.session_state.last_update = time.time()
    st.session_state.update_counter += 1
    st.rerun()
    
rng_live = np.random.default_rng(
    int(time.time() / 10)
)
st.sidebar.caption(
    f"🔄 Live Update | {time.strftime('%H:%M:%S')} | #{st.session_state.update_counter}"
)

st.sidebar.caption(
    f"🔄 Live Update alle 30s | {time.strftime('%H:%M:%S')}"
)

# =========================================================
# Custom CSS Design
# =========================================================

st.markdown("""
<style>
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


def audio_to_wav_bytes(y, sr=SR):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()


# =========================================================
# Synthetic CNC Sound Generation
# =========================================================

def generate_tool_sound(
    state,
    rpm=8000,
    teeth=4,
    material_hardness=0.7,
    machine_size=1.0,
    factory_noise=0.04,
    coolant=True,
    seed=0
):
    """
    Generates synthetic CNC tool sounds.
    The sound becomes more unstable and noisy as the tool condition worsens.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

    y = np.zeros_like(t)

    # Base machine hum
    y += 0.12 * machine_size * np.sin(2 * np.pi * 90 * t)
    y += 0.08 * machine_size * np.sin(2 * np.pi * 180 * t)
    y += 0.05 * machine_size * np.sin(2 * np.pi * 360 * t)

    # Spindle and cutting tones
    spindle_freq = rpm / 60
    tooth_pass_freq = spindle_freq * teeth

    y += 0.10 * np.sin(2 * np.pi * tooth_pass_freq * t)
    y += 0.07 * np.sin(2 * np.pi * (tooth_pass_freq * 2.1) * t)
    y += 0.04 * np.sin(2 * np.pi * (tooth_pass_freq * 3.2) * t)

    # Material hardness increases high-frequency content
    y += 0.04 * material_hardness * np.sin(2 * np.pi * (950 + 400 * material_hardness) * t)

    # Coolant pump sound
    if coolant:
        y += 0.035 * np.sin(2 * np.pi * 55 * t) * (1 + 0.4 * np.sin(2 * np.pi * 2.0 * t))

    # Factory background
    y += rng.normal(0, factory_noise, len(t))

    # Wear-specific patterns
    if state == "Healthy":
        y += rng.normal(0, 0.010, len(t))

    elif state == "Warning":
        chatter_freq = 1700 + rng.normal(0, 60)
        y += 0.08 * np.sin(2 * np.pi * chatter_freq * t)
        y += 0.05 * np.sin(2 * np.pi * (chatter_freq + 420) * t) * (1 + 0.5 * np.sin(2 * np.pi * 8 * t))
        y += rng.normal(0, 0.025, len(t))

        for _ in range(4):
            start = rng.integers(0, len(t) - 250)
            y[start:start + 250] += np.hanning(250) * rng.normal(0, 0.10, 250)

    elif state == "Critical":
        chatter_freq = 2400 + rng.normal(0, 120)
        y += 0.14 * np.sin(2 * np.pi * chatter_freq * t)
        y += 0.10 * np.sin(2 * np.pi * (chatter_freq + 620) * t)
        y += 0.08 * np.sin(2 * np.pi * 3100 * t)
        y += rng.normal(0, 0.055, len(t))

        for _ in range(10):
            start = rng.integers(0, len(t) - 320)
            y[start:start + 320] += np.hanning(320) * rng.normal(0, 0.22, 320)

    elif state == "Replace":
        chatter_freq = 3000 + rng.normal(0, 180)
        y += 0.20 * np.sin(2 * np.pi * chatter_freq * t)
        y += 0.15 * np.sin(2 * np.pi * 3800 * t)
        y += 0.13 * np.sin(2 * np.pi * 4200 * t)
        y += rng.normal(0, 0.10, len(t))

        for _ in range(22):
            start = rng.integers(0, len(t) - 280)
            y[start:start + 280] += np.hanning(280) * rng.normal(0, 0.42, 280)

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


# =========================================================
# Audio Feature Extraction
# =========================================================

def extract_features(y, sr=SR):
    """
    Extract audio features for ML classification.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    flatness = librosa.feature.spectral_flatness(y=y)

    features = []

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    for f in [centroid, bandwidth, rolloff, zcr, rms, flatness]:
        features.append(np.mean(f))
        features.append(np.std(f))

    return np.array(features)


# =========================================================
# Dataset and Model
# =========================================================

@st.cache_data
def create_training_dataset(n_per_class=70):
    X = []
    y_labels = []
    seed = 1000

    rng = np.random.default_rng(42)

    for state in CLASS_ORDER:
        for _ in range(n_per_class):
            rpm = rng.integers(5000, 14000)
            teeth = rng.choice([2, 3, 4, 5, 6])
            hardness = rng.uniform(0.35, 1.0)
            size = rng.uniform(0.7, 1.4)
            noise = rng.uniform(0.01, 0.08)
            coolant = rng.choice([True, False], p=[0.85, 0.15])

            audio = generate_tool_sound(
                state=state,
                rpm=rpm,
                teeth=teeth,
                material_hardness=hardness,
                machine_size=size,
                factory_noise=noise,
                coolant=coolant,
                seed=seed
            )

            feat = extract_features(audio)
            X.append(feat)
            y_labels.append(state)
            seed += 1

    return np.vstack(X), np.array(y_labels)


@st.cache_resource
def train_model():
    X, y = create_training_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    return model, acc, cm


# =========================================================
# Factory Generation
# =========================================================

def build_factory(n_machines=16, scenario="Normalbetrieb", seed=123):
    rng = np.random.default_rng(seed)

    cells = ["Zelle A - Fräsen", "Zelle B - Bohren", "Zelle C - Präzision", "Zelle D - Schwerzerspanung"]
    machine_types = ["3-Achs-Fräszentrum", "5-Achs-Fräszentrum", "CNC-Drehmaschine", "Bohrzentrum", "Bearbeitungszentrum"]
    tools = ["Schaftfräser", "Bohrer", "Reibahle", "Gewindebohrer", "Planfräser", "Sonderwerkzeug"]
    materials = ["Aluminium", "Stahl", "Edelstahl", "Titan", "Guss"]
    jobs = ["Gehäuse", "Welle", "Flansch", "Ventilblock", "Trägerplatte", "Motorhalter", "Hydraulikblock"]

    if scenario == "Normalbetrieb":
        probs = [0.45, 0.32, 0.17, 0.06]
    elif scenario == "Werkzeugkrise in Zelle B":
        probs = [0.25, 0.30, 0.32, 0.13]
    elif scenario == "Hohe Variantenvielfalt":
        probs = [0.35, 0.35, 0.22, 0.08]
    else:  # Störungsreicher Spätschichtbetrieb
        probs = [0.22, 0.33, 0.30, 0.15]

    rows = []

    for i in range(n_machines):
        machine = f"M{i+1:02d}"
        cell = cells[i % len(cells)]

        x = 1.5 + (i % 6) * 2.0 + rng.normal(0, 0.12)
        y = 1.5 + (i // 6) * 2.2 + rng.normal(0, 0.12)

        machine_type = rng.choice(machine_types)
        tool_type = rng.choice(tools)
        material = rng.choice(materials, p=[0.25, 0.33, 0.22, 0.10, 0.10])
        job = rng.choice(jobs)

        actual_state = rng.choice(CLASS_ORDER, p=probs)

        rpm = int(rng.integers(4500, 14500))
        teeth = int(rng.choice([2, 3, 4, 5, 6]))
        hardness_map = {
            "Aluminium": 0.35,
            "Stahl": 0.65,
            "Edelstahl": 0.78,
            "Titan": 0.95,
            "Guss": 0.70
        }

        downtime_cost = int(rng.integers(80, 240))
        due_in = int(rng.integers(25, 210))
        remaining_parts = int(rng.integers(12, 140))

        rows.append({
            "Maschine": machine,
            "Zelle": cell,
            "X": x,
            "Y": y,
            "Maschinentyp": machine_type,
            "Werkzeug_ID": f"T-{rng.integers(100, 999)}",
            "Werkzeugtyp": tool_type,
            "Material": material,
            "Auftrag": f"AUF-{rng.integers(2000, 9999)} / {job}",
            "Ist_Zustand": actual_state,
            "RPM": rpm,
            "Zähne": teeth,
            "Materialhärte": hardness_map[material],
            "Maschinengröße": rng.uniform(0.8, 1.35),
            "Stillstandskosten_EUR_min": downtime_cost,
            "Fällig_in_min": due_in,
            "Restteile": remaining_parts
        })

    return pd.DataFrame(rows)


# =========================================================
# RUL, Logistics and Decision Logic
# =========================================================

def estimate_rul(predicted_state, confidence, due_in, seed=1):
    rng = np.random.default_rng(seed)

    base = RUL_BASE[predicted_state]

    # Critical production due date may reduce allowed time window
    due_factor = 1.0
    if due_in < 45:
        due_factor = 0.90
    elif due_in > 150:
        due_factor = 1.08

    uncertainty = (1 - confidence) * 18
    noise = rng.normal(0, 4 + uncertainty)

    rul = max(1, base * due_factor + noise)
    return round(float(rul), 1)


def calculate_logistics(row, safety_margin, preset_queue, agv_queue, shortage_probability, seed=0):
    rng = np.random.default_rng(seed)

    complexity_map = {
        "Schaftfräser": 2,
        "Bohrer": 1,
        "Reibahle": 3,
        "Gewindebohrer": 2,
        "Planfräser": 3,
        "Sonderwerkzeug": 5
    }

    material_factor_map = {
        "Aluminium": 0,
        "Stahl": 2,
        "Edelstahl": 4,
        "Titan": 6,
        "Guss": 2
    }

    complexity = complexity_map[row["Werkzeugtyp"]]
    material_factor = material_factor_map[row["Material"]]

    stock_ok = rng.random() > shortage_probability

    warehouse_pick = 2 + complexity + rng.integers(0, 3)
    presetting = 7 + 3 * complexity + material_factor + preset_queue * 2 + rng.integers(0, 5)

    # Warehouse/presetting station assumed near coordinate (0,0)
    distance = np.sqrt(row["X"] ** 2 + row["Y"] ** 2)
    agv_wait = agv_queue * 2 + rng.uniform(0, 3)
    transport = 3 + distance * 1.15 + rng.uniform(0, 2)

    shortage_delay = 0
    if not stock_ok:
        shortage_delay = rng.integers(18, 38)

    total = warehouse_pick + presetting + agv_wait + transport + safety_margin + shortage_delay

    return {
        "Lager_min": round(float(warehouse_pick), 1),
        "Voreinstellung_min": round(float(presetting), 1),
        "AGV_Wartezeit_min": round(float(agv_wait), 1),
        "Transport_min": round(float(transport), 1),
        "Sicherheitsmarge_min": round(float(safety_margin), 1),
        "Bestandsverzug_min": round(float(shortage_delay), 1),
        "Logistische_Vorlaufzeit_min": round(float(total), 1),
        "Bestand_OK": bool(stock_ok)
    }


def make_decision(rul, lead_time, confidence, predicted_state, stock_ok, auto_threshold, manual_threshold):
    if not stock_ok and rul <= lead_time:
        return "BESTANDSRISIKO"

    if predicted_state == "Replace" and confidence >= 0.75 and rul <= lead_time:
        return "SOFORT_STOPP"

    if rul <= lead_time:
        if confidence >= auto_threshold:
            return "AUTO_AUFTRAG"
        elif confidence >= manual_threshold:
            return "BEDIENER_FREIGABE"
        else:
            return "UNSICHER_WARNUNG"

    if rul <= lead_time + 12:
        return "VORWARNUNG"

    return "MONITORING"


def evaluate_fleet(factory_df, model, global_noise, safety_margin, preset_queue, agv_queue,
                   shortage_probability, auto_threshold, manual_threshold, seed=999):
    rows = []
    rng = np.random.default_rng(seed)

    for idx, row in factory_df.iterrows():
        audio = generate_tool_sound(
            state=row["Ist_Zustand"],
            rpm=row["RPM"],
            teeth=row["Zähne"],
            material_hardness=row["Materialhärte"],
            machine_size=row["Maschinengröße"],
            factory_noise=global_noise,
            coolant=True,
            seed=seed + idx * 13
        )

        features = extract_features(audio).reshape(1, -1)

        pred = model.predict(features)[0]
        probas = model.predict_proba(features)[0]
        confidence = float(np.max(probas))

        rul = estimate_rul(
            predicted_state=pred,
            confidence=confidence,
            due_in=row["Fällig_in_min"],
            seed=seed + idx
        )

        logistics = calculate_logistics(
            row=row,
            safety_margin=safety_margin,
            preset_queue=preset_queue,
            agv_queue=agv_queue,
            shortage_probability=shortage_probability,
            seed=seed + idx * 7
        )

        lead = logistics["Logistische_Vorlaufzeit_min"]

        decision = make_decision(
            rul=rul,
            lead_time=lead,
            confidence=confidence,
            predicted_state=pred,
            stock_ok=logistics["Bestand_OK"],
            auto_threshold=auto_threshold,
            manual_threshold=manual_threshold
        )

        urgency_gap = lead - rul
        risk_score = (
            max(0, urgency_gap) * (row["Stillstandskosten_EUR_min"] / 100)
            + SEVERITY[pred] * 18
            + (0 if logistics["Bestand_OK"] else 22)
            + (10 if row["Fällig_in_min"] < 45 else 0)
        )

        new_row = row.to_dict()
        new_row.update({
            "KI_Zustand": pred,
            "Confidence": round(confidence, 3),
            "RUL_min": rul,
            "Entscheidung": decision,
            "Risk_Score": round(float(risk_score), 1),
        })
        new_row.update(logistics)

        rows.append(new_row)

    df = pd.DataFrame(rows)

    priority_map = {
        "SOFORT_STOPP": 1,
        "BESTANDSRISIKO": 2,
        "AUTO_AUFTRAG": 3,
        "BEDIENER_FREIGABE": 4,
        "UNSICHER_WARNUNG": 5,
        "VORWARNUNG": 6,
        "MONITORING": 7
    }

    df["Priorität_Rang"] = df["Entscheidung"].map(priority_map)
    df = df.sort_values(["Priorität_Rang", "Risk_Score"], ascending=[True, False]).reset_index(drop=True)

    return df


# =========================================================
# Plotting
# =========================================================

def plot_waveform(y, sr=SR):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    time = np.arange(len(y)) / sr
    ax.plot(time, y, color="#38bdf8", linewidth=0.9)
    ax.set_title("Waveform des Werkzeugsignals", color="white")
    ax.set_xlabel("Zeit [s]", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.25)
    return fig


def plot_spectrum(y, sr=SR):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    spectrum = np.abs(np.fft.rfft(y))
    ax.plot(freqs, spectrum, color="#a78bfa", linewidth=0.9)
    ax.set_title("Frequenzspektrum", color="white")
    ax.set_xlabel("Frequenz [Hz]", color="white")
    ax.set_ylabel("Magnitude", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, sr / 2)
    return fig


def plot_mel(y, sr=SR):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Mel-Spectrogramm", color="white")
    ax.set_xlabel("Zeitfenster", color="white")
    ax.set_ylabel("Mel-Bänder", color="white")
    ax.tick_params(colors="white")
    return fig


def factory_map(df):
    color_values = df["Entscheidung"].map(DECISION_COLORS)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["X"],
        y=df["Y"],
        mode="markers+text",
        text=df["Maschine"],
        textposition="top center",
        marker=dict(
            size=np.clip(df["Risk_Score"] * 1.1 + 16, 16, 58),
            color=color_values,
            line=dict(color="white", width=1.5),
            opacity=0.92
        ),
        customdata=np.stack([
            df["Zelle"],
            df["KI_Zustand"],
            df["RUL_min"],
            df["Logistische_Vorlaufzeit_min"],
            df["Entscheidung"],
            df["Werkzeug_ID"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Zelle: %{customdata[0]}<br>"
            "Werkzeug: %{customdata[5]}<br>"
            "KI-Zustand: %{customdata[1]}<br>"
            "RUL: %{customdata[2]} min<br>"
            "Lead Time: %{customdata[3]} min<br>"
            "Entscheidung: %{customdata[4]}<extra></extra>"
        )
    ))

    # Warehouse and presetting station
    fig.add_trace(go.Scatter(
        x=[0, 0.7],
        y=[0, 0.75],
        mode="markers+text",
        text=["Werkzeuglager", "Voreinstellung"],
        textposition="bottom center",
        marker=dict(size=[28, 26], color=["#22c55e", "#06b6d4"], symbol=["square", "diamond"]),
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Digitale Fabrikkarte – CNC-Maschinen, Werkzeuglager und Logistikrisiken",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=520,
        xaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False, title="Layout X"),
        yaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False, title="Layout Y"),
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig


def create_timeline(selected_row):
    now = pd.Timestamp.now().floor("min")

    tasks = []
    current = now

    def add_task(name, minutes, resource):
        nonlocal current
        start = current
        finish = current + pd.Timedelta(minutes=float(minutes))
        tasks.append({
            "Prozess": name,
            "Start": start,
            "Ende": finish,
            "Ressource": resource
        })
        current = finish

    add_task("KI-Erkennung & RUL-Prognose", 1, "KI-System")

    if selected_row["Bestandsverzug_min"] > 0:
        add_task("Bestandsklärung / Sonderbeschaffung", selected_row["Bestandsverzug_min"], "Werkzeuglager")

    add_task("Werkzeug im Lager reservieren", selected_row["Lager_min"], "Werkzeuglager")
    add_task("Werkzeug voreinstellen", selected_row["Voreinstellung_min"], "Voreinstellstation")
    add_task("Warten auf AGV/FTS", selected_row["AGV_Wartezeit_min"], "FTS-Leitstand")
    add_task("Transport zur Maschine", selected_row["Transport_min"], "AGV/FTS")
    add_task("Sicherheitsmarge bis Wechsel", selected_row["Sicherheitsmarge_min"], "Produktion")

    tl = pd.DataFrame(tasks)

    fig = px.timeline(
        tl,
        x_start="Start",
        x_end="Ende",
        y="Prozess",
        color="Ressource",
        title=f"Bereitstellungs-Timeline für {selected_row['Maschine']} / Werkzeug {selected_row['Werkzeug_ID']}"
    )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=420,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    fig.update_yaxes(autorange="reversed")

    return fig


# =========================================================
# KPI Simulation
# =========================================================

def simulate_kpis(n_events=180, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for method in ["Traditionell", "Predictive Tool Logistics"]:
        for _ in range(n_events):
            logistics_time = rng.uniform(20, 55)
            downtime_cost = rng.uniform(90, 240)

            if method == "Traditionell":
                notice_before_failure = rng.uniform(0, 16)
                emergency_prob = 0.72
                scrap_risk = np.clip(rng.normal(0.30, 0.10), 0, 1)
                tool_utilization = np.clip(rng.normal(0.69, 0.09), 0, 1)
            else:
                notice_before_failure = rng.uniform(28, 80)
                emergency_prob = 0.18
                scrap_risk = np.clip(rng.normal(0.11, 0.05), 0, 1)
                tool_utilization = np.clip(rng.normal(0.86, 0.05), 0, 1)

            downtime = max(0, logistics_time - notice_before_failure)
            emergency = 1 if rng.random() < emergency_prob and downtime > 0 else 0
            on_time = 1 if downtime == 0 else 0

            rows.append({
                "Methode": method,
                "Logistikzeit_min": logistics_time,
                "Erkennungszeit_vor_Ausfall_min": notice_before_failure,
                "Stillstand_min": downtime,
                "Stillstandskosten_EUR": downtime * downtime_cost,
                "Eiltransport": emergency,
                "Rechtzeitig": on_time,
                "Ausschussrisiko": scrap_risk,
                "Werkzeugausnutzung": tool_utilization
            })

    df = pd.DataFrame(rows)

    summary = df.groupby("Methode").agg(
        Gesamtstillstand_min=("Stillstand_min", "sum"),
        Durchschnittlicher_Stillstand_min=("Stillstand_min", "mean"),
        Stillstandskosten_EUR=("Stillstandskosten_EUR", "sum"),
        Eiltransporte=("Eiltransport", "sum"),
        Rechtzeitige_Bereitstellung=("Rechtzeitig", "mean"),
        Durchschnittliches_Ausschussrisiko=("Ausschussrisiko", "mean"),
        Werkzeugausnutzung=("Werkzeugausnutzung", "mean")
    ).reset_index()

    return df, summary


# =========================================================
# Header
# =========================================================

current_shift = get_current_shift()
now_str = pd.Timestamp.now().strftime("%d.%m.%Y – %H:%M:%S")

st.markdown(f"""
<div class="hero">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div style="font-size:13px; color:#93c5fd; font-weight:600; 
                        letter-spacing:2px; margin-bottom:6px;">
                ⚙️ INDUSTRIE 4.0 – PREDICTIVE MANUFACTURING
            </div>
            <div class="hero-title">
                🏭 FertigungsTech GmbH – Werk 1
            </div>
            <div class="hero-subtitle">
                Predictive Tool Logistics Control Tower | München
            </div>
            <div style="margin-top:12px; display:flex; gap:12px; flex-wrap:wrap;">
                <span style="background:rgba(34,197,94,0.2); border:1px solid #22c55e; 
                             padding:4px 12px; border-radius:999px; font-size:12px; color:#22c55e;">
                    📡 Sensor Network: AKTIV
                </span>
                <span style="background:rgba(56,189,248,0.2); border:1px solid #38bdf8; 
                             padding:4px 12px; border-radius:999px; font-size:12px; color:#38bdf8;">
                    🕐 {current_shift}
                </span>
                <span style="background:rgba(168,85,247,0.2); border:1px solid #a855f7; 
                             padding:4px 12px; border-radius:999px; font-size:12px; color:#a855f7;">
                    🔄 Live: {now_str}
                </span>
                <span style="background:rgba(245,158,11,0.2); border:1px solid #f59e0b; 
                             padding:4px 12px; border-radius:999px; font-size:12px; color:#f59e0b;">
                    ✅ ISO 9001:2015 Zertifiziert
                </span>
            </div>
        </div>
        <div style="text-align:right; color:#94a3b8; font-size:12px; min-width:200px;">
            <div style="font-size:28px; font-weight:800; color:white;">
                FTG
            </div>
            <div>FertigungsTech GmbH</div>
            <div>Industriestraße 47</div>
            <div>80339 München</div>
            <div style="margin-top:6px; color:#64748b;">
                Gegründet {FACTORY_INFO['gegruendet']} | 
                {FACTORY_INFO['mitarbeiter']} Mitarbeiter
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Sidebar Controls
# =========================================================

st.sidebar.header("⚙️ Prototype-Konfiguration")
demo_mode = st.sidebar.checkbox("🎓 Professor Demo Scenario", value=False)
priority_map = {
    "SOFORT_STOPP": 1,
    "BESTANDSRISIKO": 2,
    "AUTO_AUFTRAG": 3,
    "BEDIENER_FREIGABE": 4,
    "UNSICHER_WARNUNG": 5,
    "VORWARNUNG": 6,
    "MONITORING": 7
}

scenario = st.sidebar.selectbox(
    "Szenario",
    [
        "Normalbetrieb",
        "Werkzeugkrise in Zelle B",
        "Hohe Variantenvielfalt",
        "Störungsreicher Spätschichtbetrieb"
    ],
    index=1
)

n_machines = st.sidebar.slider("Anzahl CNC-Maschinen", 8, 24, 18, 1)

global_noise = st.sidebar.slider(
    "Fabrikgeräusch / Noise Level",
    0.01, 0.12, 0.055, 0.005
)

seed = st.sidebar.number_input(
    "Szenario Seed",
    min_value=1,
    max_value=99999,
    value=123,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Logistikparameter")

preset_queue = st.sidebar.slider("Queue Voreinstellstation", 0, 8, 2)
agv_queue = st.sidebar.slider("Queue AGV/FTS", 0, 8, 2)
safety_margin = st.sidebar.slider("Sicherheitsmarge [min]", 0, 20, 6)
shortage_probability = st.sidebar.slider("Wahrscheinlichkeit Werkzeug nicht auf Lager", 0.0, 0.35, 0.08, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("KI-Entscheidungsschwellen")

auto_threshold = st.sidebar.slider("Auto-Auftrag ab Confidence", 0.50, 1.00, 0.84, 0.01)
manual_threshold = st.sidebar.slider("Bedienerfreigabe ab Confidence", 0.30, 0.90, 0.60, 0.01)

# =========================================================
# Live Notifications System
# =========================================================

def show_notifications(fleet_df):
    sofort = fleet_df[fleet_df["Entscheidung"] == "SOFORT_STOPP"]
    auto = fleet_df[fleet_df["Entscheidung"] == "AUTO_AUFTRAG"]
    bestand = fleet_df[fleet_df["Entscheidung"] == "BESTANDSRISIKO"]
    vorwarnung = fleet_df[fleet_df["Entscheidung"] == "VORWARNUNG"]

    if len(sofort) > 0:
        for _, row in sofort.iterrows():
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg, #7f1d1d, #dc2626);
                border:2px solid #ef4444;
                border-radius:12px;
                padding:14px 18px;
                margin-bottom:8px;
                animation: pulse 1s infinite;
                display:flex;
                justify-content:space-between;
                align-items:center;">
                <div>
                    <span style="font-size:20px;">🚨</span>
                    <span style="color:white; font-weight:800; font-size:16px; margin-left:8px;">
                        SOFORT-STOPP: {row['Maschine']}
                    </span>
                    <span style="color:#fca5a5; font-size:13px; margin-left:12px;">
                        {MACHINE_REGISTRY.get(row['Maschine'], {}).get('name', '')}
                    </span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#fca5a5; font-size:13px;">
                        ⏱️ Noch {row['RUL_min']} min | 
                        Werkzeug: {row['Werkzeug_ID']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if len(bestand) > 0:
        for _, row in bestand.iterrows():
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg, #78350f, #b45309);
                border:2px solid #f59e0b;
                border-radius:12px;
                padding:14px 18px;
                margin-bottom:8px;
                display:flex;
                justify-content:space-between;
                align-items:center;">
                <div>
                    <span style="font-size:20px;">⚠️</span>
                    <span style="color:white; font-weight:800; font-size:16px; margin-left:8px;">
                        BESTANDSRISIKO: {row['Maschine']}
                    </span>
                    <span style="color:#fde68a; font-size:13px; margin-left:12px;">
                        Ersatzwerkzeug nicht verfügbar!
                    </span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#fde68a; font-size:13px;">
                        ⏱️ Noch {row['RUL_min']} min | 
                        Werkzeug: {row['Werkzeug_ID']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if len(auto) > 0:
        for _, row in auto.iterrows():
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg, #064e3b, #065f46);
                border:2px solid #10b981;
                border-radius:12px;
                padding:14px 18px;
                margin-bottom:8px;
                display:flex;
                justify-content:space-between;
                align-items:center;">
                <div>
                    <span style="font-size:20px;">📦</span>
                    <span style="color:white; font-weight:800; font-size:16px; margin-left:8px;">
                        AUTO-AUFTRAG: {row['Maschine']}
                    </span>
                    <span style="color:#6ee7b7; font-size:13px; margin-left:12px;">
                        Werkzeugbereitstellung gestartet
                    </span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#6ee7b7; font-size:13px;">
                        ⏱️ Noch {row['RUL_min']} min | 
                        Vorlaufzeit: {row['Logistische_Vorlaufzeit_min']} min
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if len(vorwarnung) > 0:
        for _, row in vorwarnung.iterrows():
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg, #1e1b4b, #3730a3);
                border:2px solid #6366f1;
                border-radius:12px;
                padding:14px 18px;
                margin-bottom:8px;
                display:flex;
                justify-content:space-between;
                align-items:center;">
                <div>
                    <span style="font-size:20px;">🔔</span>
                    <span style="color:white; font-weight:800; font-size:16px; margin-left:8px;">
                        VORWARNUNG: {row['Maschine']}
                    </span>
                    <span style="color:#a5b4fc; font-size:13px; margin-left:12px;">
                        Logistische Vorbereitung empfohlen
                    </span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#a5b4fc; font-size:13px;">
                        ⏱️ Noch {row['RUL_min']} min
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if len(sofort) == 0 and len(auto) == 0 and len(bestand) == 0:
        st.markdown("""
        <div style="background:linear-gradient(135deg, #064e3b, #065f46);
                    border:1px solid #10b981; border-radius:12px;
                    padding:12px 18px; margin-bottom:8px;">
            <span style="font-size:16px;">✅</span>
            <span style="color:#6ee7b7; font-weight:600; margin-left:8px;">
                Alle Systeme normal – Keine kritischen Alarme
            </span>
        </div>
        """, unsafe_allow_html=True)
# =========================================================
# Model Training and Fleet Evaluation
# =========================================================

with st.spinner("KI-Modell wird trainiert und Fabrikzustand wird simuliert..."):
    model, model_accuracy, cm = train_model()
    factory = build_factory(n_machines=n_machines, scenario=scenario, seed=int(seed))
    fleet = evaluate_fleet(
        factory_df=factory,
        model=model,
        global_noise=global_noise,
        safety_margin=safety_margin,
        preset_queue=preset_queue,
        agv_queue=agv_queue,
        shortage_probability=shortage_probability,
        auto_threshold=auto_threshold,
        manual_threshold=manual_threshold,
        seed=int(seed) + 700
    )
    if demo_mode:
        fleet.loc[0, "Maschine"] = "M07"
        fleet.loc[0, "Werkzeug_ID"] = "T-482"
        fleet.loc[0, "Werkzeugtyp"] = "Sonderwerkzeug"
        fleet.loc[0, "Material"] = "Titan"
        fleet.loc[0, "KI_Zustand"] = "Critical"
        fleet.loc[0, "Confidence"] = 0.91
        fleet.loc[0, "RUL_min"] = 24.0
        fleet.loc[0, "Lager_min"] = 5.0
        fleet.loc[0, "Voreinstellung_min"] = 14.0
        fleet.loc[0, "AGV_Wartezeit_min"] = 4.0
        fleet.loc[0, "Transport_min"] = 6.0
        fleet.loc[0, "Sicherheitsmarge_min"] = 5.0
        fleet.loc[0, "Bestandsverzug_min"] = 0.0
        fleet.loc[0, "Logistische_Vorlaufzeit_min"] = 34.0
        fleet.loc[0, "Bestand_OK"] = True
        fleet.loc[0, "Entscheidung"] = "AUTO_AUFTRAG"
        fleet.loc[0, "Risk_Score"] = 88.0
        fleet = fleet.sort_values(["Priorität_Rang", "Risk_Score"], ascending=[True, False]).reset_index(drop=True)
rng_live = np.random.default_rng(int(time.time() / 10))

fleet["RUL_min"] = fleet["RUL_min"].apply(
    lambda x: max(1, x + rng_live.normal(0, 1.5))
).round(1)

fleet["Risk_Score"] = fleet["Risk_Score"].apply(
    lambda x: max(0, x + rng_live.normal(0, 2.0))
).round(1)

# =========================================================
# Global KPIs
# =========================================================

urgent_count = fleet[fleet["Entscheidung"].isin(["SOFORT_STOPP", "AUTO_AUFTRAG", "BESTANDSRISIKO"])].shape[0]
manual_count = fleet[fleet["Entscheidung"].isin(["BEDIENER_FREIGABE", "UNSICHER_WARNUNG"])].shape[0]
avg_rul = fleet["RUL_min"].mean()
avg_lead = fleet["Logistische_Vorlaufzeit_min"].mean()
risk_total = fleet["Risk_Score"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    kpi_card("CNC-Maschinen", n_machines, "simulierte Fertigung", "#38bdf8")
with col2:
    kpi_card("Kritische Aufträge", urgent_count, "Auto / Stop / Bestand", "#ef4444")
with col3:
    kpi_card("Manuelle Freigaben", manual_count, "Bedienerentscheidung", "#f59e0b")
with col4:
    kpi_card("Ø RUL", f"{avg_rul:.1f} min", "Restlebensdauer", "#22c55e")
with col5:
    kpi_card("Risiko-Index", f"{risk_total:.0f}", "aggregierter Logistikdruck", "#a855f7")


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 Fertigungs-Leitwarte",
    "🔍 Maschinen & KI",
    "🚚 Logistik & Wartung",
    "📊 KPI & Berichte",
    "🧩 Architektur",
    "🤖 KI-Assistent"
])
# =========================================================
# Tab 1: Fertigungs-Leitwarte
# =========================================================

with tab1:
    st.header("🏭 Smart Factory Fertigungs-Leitwarte")

    # ============================
    # Cost Savings Live Counter
    # ============================
    elapsed_seconds = time.time() - st.session_state.start_time
    elapsed_minutes = elapsed_seconds / 60
    avg_downtime_cost = fleet["Stillstandskosten_EUR_min"].mean()
    savings_per_minute = avg_downtime_cost * 0.38
    total_savings = elapsed_minutes * savings_per_minute
    avoided_stops = int(elapsed_minutes / 45)
    avoided_transports = int(elapsed_minutes / 28)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #064e3b, #065f46);
                border:1px solid #10b981; border-radius:16px;
                padding:18px; margin-bottom:16px;">
        <div style="font-size:13px; color:#6ee7b7; 
                    letter-spacing:2px; margin-bottom:10px;">
            💰 LIVE EINSPARUNGSRECHNER – SEIT SYSTEMSTART
        </div>
        <div style="display:flex; gap:24px; flex-wrap:wrap;">
            <div>
                <div style="font-size:11px; color:#6ee7b7;">💶 Eingesparte Kosten</div>
                <div style="font-size:36px; font-weight:800; color:white;">
                    {total_savings:.2f} €
                </div>
                <div style="font-size:12px; color:#a7f3d0;">
                    seit {int(elapsed_minutes)} Minuten
                </div>
            </div>
            <div>
                <div style="font-size:11px; color:#6ee7b7;">🛑 Vermiedene Stillstände</div>
                <div style="font-size:36px; font-weight:800; color:white;">
                    {avoided_stops}
                </div>
            </div>
            <div>
                <div style="font-size:11px; color:#6ee7b7;">🚚 Vermiedene Eiltransporte</div>
                <div style="font-size:36px; font-weight:800; color:white;">
                    {avoided_transports}
                </div>
            </div>
            <div>
                <div style="font-size:11px; color:#6ee7b7;">⏱️ Systemlaufzeit</div>
                <div style="font-size:36px; font-weight:800; color:white;">
                    {int(elapsed_minutes)}m {int(elapsed_seconds % 60)}s
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================
    # Notifications
    # ============================
    st.subheader("🔔 Live Alarm Center")
    show_notifications(fleet)
    st.markdown("---")

    # ============================
    # Live Feed
    # ============================
    col_live1, col_live2, col_live3, col_live4 = st.columns(4)

    with col_live1:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.1); border:1px solid #22c55e;
                    border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:#64748b;">🏭 WERK</div>
            <div style="font-size:16px; font-weight:700; color:white;">
                {FACTORY_INFO['werk']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_live2:
        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.1); border:1px solid #38bdf8;
                    border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:#64748b;">🕐 SCHICHT</div>
            <div style="font-size:16px; font-weight:700; color:white;">
                {get_current_shift().split()[0]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_live3:
        st.markdown(f"""
        <div style="background:rgba(168,85,247,0.1); border:1px solid #a855f7;
                    border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:#64748b;">📡 SENSOREN AKTIV</div>
            <div style="font-size:16px; font-weight:700; color:white;">
                {len(fleet)} / {len(fleet)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_live4:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.1); border:1px solid #f59e0b;
                    border-radius:12px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:#64748b;">📍 STANDORT</div>
            <div style="font-size:16px; font-weight:700; color:white;">
                München, DE
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================
    # Fabrikkarte
    # ============================
    st.plotly_chart(factory_map(fleet), use_container_width=True)

    st.subheader("Priorisierte Maschinenliste")
    display_cols = [
        "Maschine", "Zelle", "Maschinentyp", "Werkzeug_ID", "Werkzeugtyp",
        "Material", "KI_Zustand", "Confidence", "RUL_min",
        "Logistische_Vorlaufzeit_min", "Entscheidung", "Risk_Score"
    ]
    st.dataframe(fleet[display_cols], use_container_width=True, height=420)

    st.subheader("Entscheidungslegende")
    legend_cols = st.columns(4)
    decisions = ["MONITORING", "VORWARNUNG", "AUTO_AUFTRAG", "BEDIENER_FREIGABE",
                 "UNSICHER_WARNUNG", "SOFORT_STOPP", "BESTANDSRISIKO"]
    for i, d in enumerate(decisions):
        with legend_cols[i % 4]:
            st.markdown(badge(d, DECISION_COLORS[d]), unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # Domino Effekt
    # ============================
    st.subheader("🔗 Domino Effekt Analyse")

    domino_machine = st.selectbox(
        "Simuliere Ausfall von Maschine:",
        fleet["Maschine"].tolist(),
        index=0,
        key="domino_select_t1"
    )

    domino_row = fleet[fleet["Maschine"] == domino_machine].iloc[0]
    machine_info_d = MACHINE_REGISTRY.get(domino_machine, {})

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #7f1d1d, #dc2626);
                border-radius:16px; padding:18px; margin:12px 0;">
        <div style="font-size:13px; color:#fca5a5; margin-bottom:4px;">
            ⚠️ SIMULIERTER MASCHINENAUSFALL
        </div>
        <div style="font-size:24px; font-weight:800; color:white;">
            {domino_machine} – {machine_info_d.get('name', '')}
        </div>
        <div style="color:#fca5a5; font-size:13px;">
            {machine_info_d.get('typ', '')} | 
            {domino_row['Zelle']} | 
            Werkzeug: {domino_row['Werkzeug_ID']} | 
            Zustand: {domino_row['KI_Zustand']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    domino_df = calculate_domino_effect(fleet, domino_machine)

    if domino_df.empty:
        st.success("✅ Keine anderen Maschinen direkt betroffen.")
    else:
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        total_cost = domino_df["Stillstandskosten_EUR_min"].sum()
        high_impact = len(domino_df[domino_df["Impact_Score"] >= 50])
        same_cell = len(domino_df[domino_df["Zelle"] == domino_row["Zelle"]])

        with col_d1:
            kpi_card("Betroffene Maschinen", len(domino_df),
                    "direkt beeinflusst", "#ef4444")
        with col_d2:
            kpi_card("Hohes Risiko", high_impact,
                    "Impact Score ≥ 50", "#f59e0b")
        with col_d3:
            kpi_card("Gleiche Zelle", same_cell,
                    "kritischste Abhängigkeit", "#7c3aed")
        with col_d4:
            kpi_card("Kosten-Risiko", f"{total_cost} €/min",
                    "bei Kettenreaktion", "#dc2626")

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.dataframe(
                domino_df[["Maschine", "Zelle", "KI_Zustand",
                           "RUL_min", "Impact_Score", "Gründe"]],
                use_container_width=True,
                height=300,
                hide_index=True
            )
        with col_right:
            fig_domino = px.bar(
                domino_df.head(10),
                x="Maschine",
                y="Impact_Score",
                color="KI_Zustand",
                color_discrete_map=STATE_COLORS,
                title="Domino Impact Score",
                text="Impact_Score"
            )
            fig_domino.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                font=dict(color="white"),
                height=300
            )
            st.plotly_chart(fig_domino, use_container_width=True)
# =========================================================
# Tab 2: Maschinen und KI
# =========================================================

with tab2:
    st.header("🔍 Maschinen & KI Analyse")

    selected_machine = st.selectbox(
        "Maschine auswählen",
        fleet["Maschine"].tolist(),
        index=0
    )

    selected = fleet[fleet["Maschine"] == selected_machine].iloc[0]
    machine_info = MACHINE_REGISTRY.get(selected_machine, {})
    sensor_data = get_sensor_reading(
        machine_id=selected_machine,
        state=selected["Ist_Zustand"],
        rpm=selected["RPM"],
        seed=int(seed) + hash(selected_machine) % 1000
    )

    # KPI Cards
    colA, colB, colC, colD = st.columns(4)
    with colA:
        kpi_card("Maschine", selected["Maschine"],
                selected["Maschinentyp"], "#38bdf8")
    with colB:
        kpi_card("Werkzeug", selected["Werkzeug_ID"],
                selected["Werkzeugtyp"], "#22c55e")
    with colC:
        kpi_card("KI-Zustand", selected["KI_Zustand"],
                f"Confidence {selected['Confidence']*100:.1f}%",
                STATE_COLORS[selected["KI_Zustand"]])
    with colD:
        kpi_card("Entscheidung", selected["Entscheidung"],
                f"Risk Score {selected['Risk_Score']}",
                DECISION_COLORS[selected["Entscheidung"]])

    st.markdown("---")

    # Maschinen-Identität
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                border:1px solid #334155; border-radius:16px; 
                padding:18px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; 
                    align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:11px; color:#64748b; 
                            letter-spacing:2px; margin-bottom:4px;">
                    MASCHINEN-IDENTITÄT
                </div>
                <div style="font-size:22px; font-weight:800; color:white;">
                    {machine_info.get('name', selected_machine)}
                </div>
                <div style="color:#94a3b8; font-size:13px;">
                    {machine_info.get('typ', '')} | 
                    {machine_info.get('zelle', '')} | 
                    Baujahr {machine_info.get('baujahr', '')}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#64748b; margin-bottom:4px;">
                    SENSOR ID
                </div>
                <div style="font-family:monospace; color:#38bdf8; font-size:14px;">
                    {sensor_data['sensor_id']}
                </div>
                <div style="color:#94a3b8; font-size:12px; margin-top:4px;">
                    Messung #{sensor_data['messung_nr']}
                </div>
                <div style="color:#64748b; font-size:11px;">
                    {sensor_data['timestamp']}
                </div>
            </div>
        </div>
        <div style="margin-top:16px; display:flex; gap:10px; flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.05); border-radius:10px; 
                        padding:10px 16px; min-width:120px;">
                <div style="font-size:11px; color:#64748b;">🌡️ Temperatur</div>
                <div style="font-size:20px; font-weight:700; color:#f59e0b;">
                    {sensor_data['temperatur_c']}°C
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:10px; 
                        padding:10px 16px; min-width:120px;">
                <div style="font-size:11px; color:#64748b;">📳 Vibration</div>
                <div style="font-size:20px; font-weight:700; color:#a855f7;">
                    {sensor_data['vibration_mm_s']} mm/s
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:10px; 
                        padding:10px 16px; min-width:120px;">
                <div style="font-size:11px; color:#64748b;">⚡ Spindelstrom</div>
                <div style="font-size:20px; font-weight:700; color:#38bdf8;">
                    {sensor_data['spindelstrom_a']} A
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:10px; 
                        padding:10px 16px; min-width:120px;">
                <div style="font-size:11px; color:#64748b;">🔊 Audio RMS</div>
                <div style="font-size:20px; font-weight:700; color:#22c55e;">
                    {sensor_data['audio_rms']}
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:10px; 
                        padding:10px 16px; min-width:120px;">
                <div style="font-size:11px; color:#64748b;">❄️ Kühlmittel</div>
                <div style="font-size:20px; font-weight:700; color:#06b6d4;">
                    {sensor_data['kuehlmittel_temp_c']}°C
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:10px; 
                        padding:10px 16px; min-width:120px;">
                <div style="font-size:11px; color:#64748b;">👤 Bediener</div>
                <div style="font-size:16px; font-weight:700; color:white;">
                    {machine_info.get('bediener', '-')}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge Charts
    st.subheader("📈 Live Gauge Monitor")
    sensor_live = get_sensor_reading(
        machine_id=selected_machine,
        state=selected["Ist_Zustand"],
        rpm=selected["RPM"],
        seed=int(time.time() / 5)
    )

    g1, g2, g3, g4 = st.columns(4)

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

    with g1:
        st.plotly_chart(
            make_gauge("🌡️ Temperatur", sensor_live["temperatur_c"],
                      0, 120, "°C", "#f59e0b"),
            use_container_width=True
        )
    with g2:
        st.plotly_chart(
            make_gauge("📳 Vibration", sensor_live["vibration_mm_s"],
                      0, 3.0, " mm/s", "#a855f7"),
            use_container_width=True
        )
    with g3:
        st.plotly_chart(
            make_gauge("⚡ Spindelstrom", sensor_live["spindelstrom_a"],
                      0, 25, " A", "#38bdf8"),
            use_container_width=True
        )
    with g4:
        st.plotly_chart(
            make_gauge("⏱️ RUL", selected["RUL_min"],
                      0, 100, " min", "#22c55e"),
            use_container_width=True
        )

    st.markdown("---")

    # Produktionsauftrag & Logistik
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Produktionsauftrag")
        info_df = pd.DataFrame({
            "Parameter": ["Zelle", "Auftrag", "Material", "RPM",
                         "Zähne", "Restteile", "Fällig in", "Stillstandskosten"],
            "Wert": [
                selected["Zelle"], selected["Auftrag"],
                selected["Material"], selected["RPM"],
                selected["Zähne"], selected["Restteile"],
                f"{selected['Fällig_in_min']} min",
                f"{selected['Stillstandskosten_EUR_min']} €/min"
            ]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Werkzeuglogistische Bewertung")
        logistics_df = pd.DataFrame({
            "Kennzahl": ["RUL", "Logistische Vorlaufzeit", "Lagerentnahme",
                        "Voreinstellung", "AGV-Wartezeit", "Transport",
                        "Sicherheitsmarge", "Bestandsverzug", "Bestand OK"],
            "Wert": [
                f"{selected['RUL_min']} min",
                f"{selected['Logistische_Vorlaufzeit_min']} min",
                f"{selected['Lager_min']} min",
                f"{selected['Voreinstellung_min']} min",
                f"{selected['AGV_Wartezeit_min']} min",
                f"{selected['Transport_min']} min",
                f"{selected['Sicherheitsmarge_min']} min",
                f"{selected['Bestandsverzug_min']} min",
                "Ja" if selected["Bestand_OK"] else "Nein"
            ]
        })
        st.dataframe(logistics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Decision Status
    decision = selected["Entscheidung"]
    if decision == "AUTO_AUFTRAG":
        st.success("✅ Automatischer Werkzeugbereitstellungsauftrag wurde erzeugt.")
    elif decision == "SOFORT_STOPP":
        st.error("🚨 Sofortiger Stopp empfohlen!")
    elif decision == "BEDIENER_FREIGABE":
        st.warning("⚠️ Bedienerfreigabe erforderlich.")
    elif decision == "BESTANDSRISIKO":
        st.error("❌ Bestandsrisiko: Ersatzwerkzeug nicht verfügbar!")
    elif decision == "VORWARNUNG":
        st.info("🔔 Vorwarnung: Logistische Vorbereitung empfohlen.")
    else:
        st.info("✅ Monitoring: Werkzeug kann weiterlaufen.")

    st.markdown("---")

    # Audio & KI
    st.subheader("🎧 Akustische Analyse & KI")

    selected_audio_machine = st.selectbox(
        "Maschine für Audioanalyse",
        fleet["Maschine"].tolist(),
        index=0,
        key="audio_machine_t2"
    )

    audio_row = fleet[fleet["Maschine"] == selected_audio_machine].iloc[0]
    audio = generate_tool_sound(
        state=audio_row["Ist_Zustand"],
        rpm=audio_row["RPM"],
        teeth=audio_row["Zähne"],
        material_hardness=audio_row["Materialhärte"],
        machine_size=audio_row["Maschinengröße"],
        factory_noise=global_noise,
        coolant=True,
        seed=int(seed) + int(audio_row.name) * 13 + 1000
    )

    st.audio(audio_to_wav_bytes(audio), format="audio/wav")

    # Live Signal
    if st.button("🔄 Signal aktualisieren", key="refresh_t2"):
        st.rerun()

    live_seed = int(time.time() / 5)
    rng_signal = np.random.default_rng(live_seed)
    window_start = int(rng_signal.integers(0, len(audio) - SR))
    window = audio[window_start:window_start + SR]
    time_axis = np.linspace(0, 1, len(window))

    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=time_axis, y=window, mode="lines",
        line=dict(color=STATE_COLORS[audio_row["KI_Zustand"]], width=1.2)
    ))
    fig_live.add_hline(y=0.5, line_dash="dash", line_color="#ef4444",
                       annotation_text="⚠️ Kritische Schwelle",
                       annotation_font_color="#ef4444")
    fig_live.add_hline(y=-0.5, line_dash="dash", line_color="#ef4444")
    fig_live.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#0f172a",
        font=dict(color="white"), height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(
            text=f"🔴 LIVE | {audio_row['Maschine']} | {audio_row['KI_Zustand']}",
            font=dict(color="white", size=13)
        )
    )
    st.plotly_chart(fig_live, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_waveform(audio))
        st.pyplot(plot_spectrum(audio))
    with col2:
        st.pyplot(plot_mel(audio))

        features = extract_features(audio).reshape(1, -1)
        probas = model.predict_proba(features)[0]
        proba_df = pd.DataFrame({
            "Zustand": model.classes_,
            "Wahrscheinlichkeit": probas
        })
        fig_prob = px.bar(
            proba_df, x="Zustand", y="Wahrscheinlichkeit",
            color="Zustand", color_discrete_map=STATE_COLORS,
            title="KI-Wahrscheinlichkeiten", text="Wahrscheinlichkeit"
        )
        fig_prob.update_traces(
            texttemplate="%{text:.2f}", textposition="outside"
        )
        fig_prob.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#0f172a",
            font=dict(color="white"), yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # KI Kennzahlen
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Tatsächlicher Zustand", audio_row["Ist_Zustand"],
                "simuliert", STATE_COLORS[audio_row["Ist_Zustand"]])
    with c2:
        kpi_card("KI-Zustand", audio_row["KI_Zustand"],
                "Vorhersage", STATE_COLORS[audio_row["KI_Zustand"]])
    with c3:
        kpi_card("Confidence", f"{audio_row['Confidence']*100:.1f}%",
                "Modellsicherheit", "#38bdf8")
    with c4:
        kpi_card("Test Accuracy", f"{model_accuracy*100:.1f}%",
                "synthetische Testdaten", "#a855f7")

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")

    # Digital Twin Soundtrack
    st.subheader("🎵 Digital Twin Soundtrack – Akustischer Fingerabdruck")
    st.info("""
    Jede CNC-Maschine hat eine einzigartige akustische Signatur.
    Das System vergleicht das aktuelle Signal mit der Baseline.
    """)

    baseline_audio = generate_tool_sound(
        state="Healthy", rpm=audio_row["RPM"],
        teeth=audio_row["Zähne"],
        material_hardness=audio_row["Materialhärte"],
        machine_size=audio_row["Maschinengröße"],
        factory_noise=0.02, coolant=True,
        seed=int(seed) + 9999
    )

    baseline_features = learn_acoustic_fingerprint(
        audio_row["Maschine"], baseline_audio
    )
    current_features = extract_features(audio)
    deviation_pct, deviation_status = compare_to_fingerprint(
        audio_row["Maschine"], current_features
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        kpi_card("Akustische Abweichung", f"{deviation_pct}%",
                deviation_status,
                "#ef4444" if deviation_pct > 35 else "#22c55e")
    with col_f2:
        kpi_card("Baseline Status", "✅ Gelernt",
                f"Maschine {audio_row['Maschine']}", "#38bdf8")
    with col_f3:
        kpi_card("Fingerabdruck", "Aktiv",
                f"seit {pd.Timestamp.now().strftime('%H:%M')}", "#a855f7")
# =========================================================
# Tab 3: Logistik und Wartung
# =========================================================
with tab3:
    st.header("🚚 Logistik & Wartung")

    # ============================
    # Logistik-Leitstand
    # ============================
    st.subheader("🚚 Logistik-Leitstand")

    order_filter = fleet[fleet["Entscheidung"].isin([
        "SOFORT_STOPP", "BESTANDSRISIKO", "AUTO_AUFTRAG",
        "BEDIENER_FREIGABE", "UNSICHER_WARNUNG", "VORWARNUNG"
    ])].copy()

    if order_filter.empty:
        st.info("Aktuell keine Werkzeugbereitstellungsaufträge notwendig.")
    else:
        st.subheader("Order Board")
        st.dataframe(
            order_filter[[
                "Maschine", "Zelle", "Werkzeug_ID", "Werkzeugtyp",
                "KI_Zustand", "RUL_min", "Logistische_Vorlaufzeit_min",
                "Entscheidung", "Bestand_OK", "Risk_Score"
            ]],
            use_container_width=True, height=300
        )

    st.markdown("---")

    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("🤖 AGV / FTS Status")
        critical_orders = order_filter.head(4)
        agv_rows = []
        for i in range(1, 5):
            if i <= len(critical_orders):
                r = critical_orders.iloc[i-1]
                status = f"Transport für {r['Maschine']}"
                target = r["Zelle"]
                load = r["Werkzeug_ID"]
            else:
                status = "Bereit"
                target = "-"
                load = "-"
            agv_rows.append({
                "AGV": f"FTS-{i:02d}",
                "Status": status,
                "Ziel": target,
                "Ladung": load
            })
        st.dataframe(
            pd.DataFrame(agv_rows),
            use_container_width=True,
            hide_index=True
        )

    with colR:
        st.subheader("🔧 Voreinstellstationen")
        preset_rows = []
        for i in range(1, 4):
            if i <= len(order_filter):
                r = order_filter.iloc[i-1]
                status = f"Voreinstellung {r['Werkzeug_ID']}"
                machine = r["Maschine"]
                remaining = f"{r['Voreinstellung_min']} min"
            else:
                status = "Frei"
                machine = "-"
                remaining = "-"
            preset_rows.append({
                "Station": f"Preset-{i}",
                "Status": status,
                "Für Maschine": machine,
                "Restzeit": remaining
            })
        st.dataframe(
            pd.DataFrame(preset_rows),
            use_container_width=True,
            hide_index=True
        )

    if not order_filter.empty:
        st.markdown("---")
        timeline_machine = st.selectbox(
            "Timeline für Auftrag anzeigen",
            order_filter["Maschine"].tolist(),
            key="timeline_t3"
        )
        timeline_row = order_filter[
            order_filter["Maschine"] == timeline_machine
        ].iloc[0]
        st.plotly_chart(
            create_timeline(timeline_row),
            use_container_width=True
        )

        actions = pd.DataFrame({
            "Aktion": [
                "Werkzeugzustand erkannt",
                "Ersatzwerkzeug prüfen",
                "Werkzeug reservieren",
                "Voreinstellauftrag erzeugen",
                "Transportauftrag an FTS senden",
                "Werkzeug an Maschine liefern",
                "Wechselzeitpunkt vorschlagen"
            ],
            "System": [
                "KI-Modul", "Werkzeuglager",
                "WMS / Tool Management", "Voreinstellstation",
                "FTS-Leitsystem", "AGV / Mitarbeiter",
                "MES / Bedienerinterface"
            ],
            "Status": [
                "Abgeschlossen",
                "Abgeschlossen" if timeline_row["Bestand_OK"] else "Problem",
                "Gestartet", "Geplant", "Geplant",
                "In Vorbereitung", "Offen"
            ]
        })
        st.dataframe(actions, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ============================
    # Wartungskalender
    # ============================
    st.subheader("📅 Predictiver Wartungskalender")

    now = pd.Timestamp.now()
    calendar_data = []

    for _, row in fleet.iterrows():
        machine_info = MACHINE_REGISTRY.get(row["Maschine"], {})
        maintenance_time = now + pd.Timedelta(minutes=float(row["RUL_min"]))

        if row["RUL_min"] < 30:
            priority = "🚨 Kritisch"
            color = "#ef4444"
        elif row["RUL_min"] < 60:
            priority = "⚠️ Dringend"
            color = "#f59e0b"
        elif row["RUL_min"] < 90:
            priority = "🔔 Bald"
            color = "#6366f1"
        else:
            priority = "✅ Normal"
            color = "#22c55e"

        calendar_data.append({
            "Maschine": row["Maschine"],
            "Name": machine_info.get("name", ""),
            "Zelle": row["Zelle"],
            "Werkzeug": row["Werkzeug_ID"],
            "RUL_min": row["RUL_min"],
            "Wartung_um": maintenance_time.strftime("%d.%m.%Y %H:%M"),
            "Priorität": priority,
            "Farbe": color,
            "Bediener": machine_info.get("bediener", "-")
        })

    cal_df = pd.DataFrame(calendar_data).sort_values(
        "RUL_min", ascending=True
    ).reset_index(drop=True)

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        kpi_card("🚨 Kritisch",
                len(cal_df[cal_df["RUL_min"] < 30]),
                "< 30 min", "#ef4444")
    with col_k2:
        kpi_card("⚠️ Dringend",
                len(cal_df[(cal_df["RUL_min"] >= 30) &
                           (cal_df["RUL_min"] < 60)]),
                "30-60 min", "#f59e0b")
    with col_k3:
        kpi_card("🔔 Bald",
                len(cal_df[(cal_df["RUL_min"] >= 60) &
                           (cal_df["RUL_min"] < 90)]),
                "60-90 min", "#6366f1")
    with col_k4:
        kpi_card("✅ Normal",
                len(cal_df[cal_df["RUL_min"] >= 90]),
                "> 90 min", "#22c55e")

    fig_cal = go.Figure()
    for i, row in cal_df.iterrows():
        fig_cal.add_trace(go.Bar(
            x=[row["RUL_min"]],
            y=[f"{row['Maschine']} – {row['Name']}"],
            orientation="h",
            marker=dict(color=row["Farbe"]),
            text=f"{row['RUL_min']} min | {row['Wartung_um']}",
            textposition="inside",
            showlegend=False
        ))

    fig_cal.add_vline(x=30, line_dash="dash", line_color="#ef4444",
                      annotation_text="🚨 Kritisch",
                      annotation_font_color="#ef4444")
    fig_cal.add_vline(x=60, line_dash="dash", line_color="#f59e0b",
                      annotation_text="⚠️ Dringend",
                      annotation_font_color="#f59e0b")

    fig_cal.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#0f172a",
        font=dict(color="white"), height=500,
        xaxis=dict(title="Verbleibende Zeit [min]",
                  gridcolor="#334155", color="white"),
        yaxis=dict(gridcolor="#334155", color="white"),
        title=dict(
            text="Wartungsplan – Alle Maschinen nach RUL sortiert",
            font=dict(color="white")
        )
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.dataframe(
        cal_df[["Priorität", "Maschine", "Name", "Zelle",
                "Werkzeug", "RUL_min", "Wartung_um", "Bediener"]],
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

    # ============================
    # Gabelstapler Wartung
    # ============================
    st.subheader("🚜 Gabelstapler Predictive Maintenance")

    st.markdown("""
    <div style="background:linear-gradient(135deg, #1e3a8a, #1e293b);
                border:1px solid #3b82f6; border-radius:12px; padding:14px;
                margin-bottom:16px;">
        <div style="color:#93c5fd; font-size:13px;">
            🚜 Das System überwacht auch die Gabelstapler im Werk und 
            sagt voraus, wann Batterie, Reifen oder Motor gewartet werden müssen.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # بيانات الرافعات الشوكية
    GABELSTAPLER = {
        "G01": {"name": "Linde E25", "typ": "Elektro",
                "bediener": "K. Weber", "baujahr": 2020},
        "G02": {"name": "Toyota 8FBE20", "typ": "Elektro",
                "bediener": "T. Müller", "baujahr": 2019},
        "G03": {"name": "Still RX20-20", "typ": "Elektro",
                "bediener": "F. Schmidt", "baujahr": 2021},
        "G04": {"name": "Jungheinrich EFG425", "typ": "Elektro",
                "bediener": "R. Bauer", "baujahr": 2018},
        "G05": {"name": "Linde H25D", "typ": "Diesel",
                "bediener": "M. Klein", "baujahr": 2019},
    }

    rng_g = np.random.default_rng(int(seed) + 500)

    gabel_data = []
    for g_id, g_info in GABELSTAPLER.items():
        batterie_rul = round(float(rng_g.uniform(10, 95)), 1)
        reifen_rul = round(float(rng_g.uniform(20, 200)), 1)
        motor_rul = round(float(rng_g.uniform(50, 300)), 1)
        oel_rul = round(float(rng_g.uniform(30, 150)), 1)

        if batterie_rul < 20:
            status = "🚨 Kritisch"
            color = "#ef4444"
        elif batterie_rul < 40:
            status = "⚠️ Dringend"
            color = "#f59e0b"
        else:
            status = "✅ Normal"
            color = "#22c55e"

        gabel_data.append({
            "Stapler": g_id,
            "Name": g_info["name"],
            "Typ": g_info["typ"],
            "Bediener": g_info["bediener"],
            "🔋 Batterie RUL [h]": batterie_rul,
            "🛞 Reifen RUL [h]": reifen_rul,
            "⚙️ Motor RUL [h]": motor_rul,
            "🛢️ Öl RUL [h]": oel_rul,
            "Status": status,
            "Farbe": color
        })

    gabel_df = pd.DataFrame(gabel_data)

    # KPIs
    g_krit = len(gabel_df[gabel_df["Status"] == "🚨 Kritisch"])
    g_drng = len(gabel_df[gabel_df["Status"] == "⚠️ Dringend"])
    g_norm = len(gabel_df[gabel_df["Status"] == "✅ Normal"])

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        kpi_card("🚨 Kritisch", g_krit, "sofort warten", "#ef4444")
    with gc2:
        kpi_card("⚠️ Dringend", g_drng, "bald warten", "#f59e0b")
    with gc3:
        kpi_card("✅ Normal", g_norm, "kein Handlungsbedarf", "#22c55e")

    st.dataframe(
        gabel_df[[
            "Stapler", "Name", "Typ", "Bediener",
            "🔋 Batterie RUL [h]", "🛞 Reifen RUL [h]",
            "⚙️ Motor RUL [h]", "🛢️ Öl RUL [h]", "Status"
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Chart
    fig_gabel = go.Figure()

    komponenten = ["🔋 Batterie RUL [h]", "🛞 Reifen RUL [h]",
                   "⚙️ Motor RUL [h]", "🛢️ Öl RUL [h]"]
    colors_g = ["#38bdf8", "#a855f7", "#22c55e", "#f59e0b"]

    for comp, col in zip(komponenten, colors_g):
        fig_gabel.add_trace(go.Bar(
            name=comp,
            x=gabel_df["Stapler"],
            y=gabel_df[comp],
            marker_color=col
        ))

    fig_gabel.update_layout(
        barmode="group",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=400,
        title=dict(
            text="Gabelstapler – Verbleibende Nutzungszeit pro Komponente",
            font=dict(color="white")
        ),
        xaxis=dict(gridcolor="#334155", color="white"),
        yaxis=dict(
            title="RUL [Stunden]",
            gridcolor="#334155",
            color="white"
        )
    )

    st.plotly_chart(fig_gabel, use_container_width=True)

# =========================================================
# Tab 4: KI und Berichte
# =========================================================
with tab4:
    st.header("📊 KPI & Berichte")

    # ============================
    # KPI Simulation
    # ============================
    st.subheader("📊 KPI Simulation: Traditionell vs. Predictive")

    kpi_events = st.slider(
        "Anzahl simulierter Werkzeugereignisse", 50, 500, 180, 10
    )

    kpi_df, summary = simulate_kpis(
        n_events=kpi_events, seed=int(seed) + 400
    )

    st.dataframe(summary, use_container_width=True)

    traditional = summary[summary["Methode"] == "Traditionell"].iloc[0]
    predictive = summary[
        summary["Methode"] == "Predictive Tool Logistics"
    ].iloc[0]

    downtime_reduction = (
        1 - predictive["Gesamtstillstand_min"] /
        traditional["Gesamtstillstand_min"]
    ) * 100

    emergency_reduction = (
        1 - predictive["Eiltransporte"] /
        max(traditional["Eiltransporte"], 1)
    ) * 100

    cost_reduction = (
        1 - predictive["Stillstandskosten_EUR"] /
        traditional["Stillstandskosten_EUR"]
    ) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Stillstandsreduktion", f"{downtime_reduction:.1f}%",
                "gegenüber traditionell", "#22c55e")
    with col2:
        kpi_card("Eiltransport-Reduktion", f"{emergency_reduction:.1f}%",
                "weniger Störungen", "#38bdf8")
    with col3:
        kpi_card("Kostenreduktion", f"{cost_reduction:.1f}%",
                "simulierte Stillstandskosten", "#a855f7")

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        fig1 = px.bar(summary, x="Methode", y="Gesamtstillstand_min",
                     color="Methode",
                     title="Gesamter Maschinenstillstand [min]")
        fig1.update_layout(paper_bgcolor="#111827",
                          plot_bgcolor="#0f172a",
                          font=dict(color="white"))
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.bar(summary, x="Methode", y="Eiltransporte",
                     color="Methode", title="Anzahl Eiltransporte")
        fig2.update_layout(paper_bgcolor="#111827",
                          plot_bgcolor="#0f172a",
                          font=dict(color="white"))
        st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig3 = px.bar(summary, x="Methode",
                     y="Rechtzeitige_Bereitstellung",
                     color="Methode",
                     title="Anteil rechtzeitiger Bereitstellungen")
        fig3.update_layout(paper_bgcolor="#111827",
                          plot_bgcolor="#0f172a",
                          font=dict(color="white"))
        st.plotly_chart(fig3, use_container_width=True)

    with colD:
        fig4 = px.bar(summary, x="Methode",
                     y="Durchschnittliches_Ausschussrisiko",
                     color="Methode",
                     title="Durchschnittliches Ausschussrisiko")
        fig4.update_layout(paper_bgcolor="#111827",
                          plot_bgcolor="#0f172a",
                          font=dict(color="white"))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ============================
    # Schichtbericht
    # ============================
    st.subheader("📄 Automatischer Schichtbericht")

    now = pd.Timestamp.now()
    current_shift = get_current_shift()

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        kpi_card("Aktuelle Schicht", current_shift.split()[0],
                current_shift, "#38bdf8")
    with col_s2:
        kpi_card("Datum", now.strftime("%d.%m.%Y"),
                now.strftime("%H:%M:%S Uhr"), "#22c55e")
    with col_s3:
        kpi_card("Werk", "Werk 1 – München",
                "FertigungsTech GmbH", "#a855f7")

    sofort_count = len(fleet[fleet["Entscheidung"] == "SOFORT_STOPP"])
    auto_count = len(fleet[fleet["Entscheidung"] == "AUTO_AUFTRAG"])
    bestand_count = len(fleet[fleet["Entscheidung"] == "BESTANDSRISIKO"])
    vorwarnung_count = len(fleet[fleet["Entscheidung"] == "VORWARNUNG"])
    monitoring_count = len(fleet[fleet["Entscheidung"] == "MONITORING"])

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        kpi_card("🚨 Sofort-Stopps", sofort_count,
                "kritische Eingriffe", "#ef4444")
    with col_b:
        kpi_card("📦 Auto-Aufträge", auto_count,
                "automatisch ausgelöst", "#22c55e")
    with col_c:
        kpi_card("⚠️ Bestandsrisiken", bestand_count,
                "Lagerproblem erkannt", "#f59e0b")
    with col_d:
        kpi_card("✅ Monitoring", monitoring_count,
                "Normalbetrieb", "#38bdf8")

    st.markdown("---")

    elapsed_minutes = (
        time.time() - st.session_state.start_time
    ) / 60
    avg_downtime_cost = fleet["Stillstandskosten_EUR_min"].mean()
    total_savings = elapsed_minutes * avg_downtime_cost * 0.38
    avg_confidence = fleet["Confidence"].mean()
    total_risk = fleet["Risk_Score"].sum()

    kritische = fleet[fleet["Entscheidung"].isin([
        "SOFORT_STOPP", "AUTO_AUFTRAG", "BESTANDSRISIKO"
    ])]

    kritische_liste = ""
    for _, row in kritische.iterrows():
        machine_name = MACHINE_REGISTRY.get(
            row["Maschine"], {}
        ).get("name", "")
        kritische_liste += (
            f"• {row['Maschine']} ({machine_name}): "
            f"{row['Entscheidung']} – RUL: {row['RUL_min']} min\n"
        )

    bericht_text = f"""
SCHICHTBERICHT – FertigungsTech GmbH – Werk 1, München
{"="*55}
Datum:           {now.strftime("%d.%m.%Y")}
Uhrzeit:         {now.strftime("%H:%M:%S")} Uhr
Schicht:         {current_shift}
Erstellt von:    Predictive Tool Logistics System (KI)
{"="*55}

ZUSAMMENFASSUNG
---------------
Überwachte Maschinen:     {len(fleet)}
Sofort-Stopps:            {sofort_count}
Automatische Aufträge:    {auto_count}
Bestandsrisiken:          {bestand_count}
Vorwarnungen:             {vorwarnung_count}
Normalbetrieb:            {monitoring_count}

KI-KENNZAHLEN
-------------
Durchschnittliche RUL:    {fleet['RUL_min'].mean():.1f} min
Durchschnittl. Confidence:{avg_confidence*100:.1f}%
Gesamt-Risiko-Index:      {total_risk:.0f}

WIRTSCHAFTLICHE KENNZAHLEN
---------------------------
Systembetrieb:            {int(elapsed_minutes)} Minuten
Eingesparte Kosten:       {total_savings:.2f} €
Vermiedene Stillstände:   {int(elapsed_minutes/45)}
Vermiedene Eiltransporte: {int(elapsed_minutes/28)}

KRITISCHE MASCHINEN
-------------------
{kritische_liste if kritische_liste else "Keine kritischen Maschinen."}

EMPFEHLUNGEN
------------
{"• Sofortige Überprüfung erforderlich!" if sofort_count > 0 else "• Normaler Betrieb kann fortgesetzt werden"}
{"• Werkzeugbestand für " + str(bestand_count) + " Maschinen prüfen" if bestand_count > 0 else ""}
{"• " + str(auto_count) + " Werkzeuge automatisch bestellt" if auto_count > 0 else ""}

{"="*55}
SYSTEM: Predictive Tool Logistics – FertigungsTech GmbH
KI-Modell: Random Forest + Gemini AI
{"="*55}
    """

    st.code(bericht_text, language="text")

    st.download_button(
        label="📥 Schichtbericht herunterladen",
        data=bericht_text,
        file_name=f"Schichtbericht_{now.strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

# =========================================================
# Tab 5: Architektur
# =========================================================
with tab5:
    st.header("🧩 Architektur, Innovationskern und Präsentations-Pitch")

    st.subheader("Systemarchitektur")

    st.code("""
Akustische Werkzeugsignale (CNC-Maschinen)
        ↓
Audio Feature Extraction (Librosa)
        ↓
KI-Modell zur Werkzeugzustandserkennung (Random Forest)
        ↓
Restlebensdauer-Prognose RUL
        ↓
Vergleich mit logistischer Vorlaufzeit
        ↓
Automatische Entscheidung:
    - Monitoring
    - Vorwarnung
    - Bedienerfreigabe
    - Automatischer Bereitstellungsauftrag
    - Sofort-Stopp
        ↓
Werkzeuglager → Voreinstellstation → AGV/FTS → CNC-Maschine
        ↓
Gabelstapler Predictive Maintenance (Erweiterung)
    """, language="text")

    st.subheader("Innovationskern")

    st.success("""
    Der Prototyp verbindet akustische KI-Erkennung mit automatisierter 
    Produktionslogistik. Er wandelt Sensordaten direkt in logistische 
    Entscheidungen um: Wann muss ein Ersatzwerkzeug vorbereitet werden?
    Zusätzlich überwacht das System Gabelstapler und ermöglicht damit
    eine vollständige Fabriklogistik-Überwachung.
    """)

    st.subheader("Industrie 4.0 Bausteine")

    ind40 = pd.DataFrame({
        "Industrie-4.0-Baustein": [
            "IoT / Sensorik",
            "KI / Machine Learning",
            "Predictive Analytics",
            "Produktionslogistik",
            "Automatisierung",
            "Digital Twin",
            "Smart Factory",
            "Generative KI"
        ],
        "Umsetzung im Prototyp": [
            "Akustische Werkzeugsignale + Sensor-Daten",
            "Random Forest Klassifikation",
            "RUL-Prognose + Gabelstapler Wartung",
            "Werkzeuglager, Voreinstellung, AGV/FTS",
            "Automatische Bereitstellungsentscheidung",
            "Digital Twin Soundtrack – Akustischer Fingerabdruck",
            "Control Tower mit Live-Daten und Domino-Effekt",
            "Gemini AI Chat Assistant"
        ]
    })

    st.dataframe(ind40, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Präsentations-Pitch")

    st.info("""
    Unser Projekt „Predictive Tool Logistics" verbindet KI-basierte 
    akustische Werkzeugzustandserkennung mit automatisierter Logistik.
    
    Das System analysiert CNC-Werkzeugsignale, prognostiziert den 
    Werkzeugzustand und vergleicht die Restlebensdauer mit der 
    logistischen Vorlaufzeit. Wenn die Zeit nicht mehr ausreicht, 
    wird automatisch ein Bereitstellungsauftrag erzeugt.
    
    Zusätzlich überwacht das System Gabelstapler im Werk und sagt 
    voraus, wann Batterie, Reifen oder Motor gewartet werden müssen.
    
    Dadurch werden ungeplante Stillstände, Eiltransporte und 
    Ausschussrisiken deutlich reduziert.
    """)

    st.subheader("Risiken und Gegenmaßnahmen")

    risks = pd.DataFrame({
        "Risiko": [
            "Reale Fabrikgeräusche stören Audioanalyse",
            "KI braucht echte Trainingsdaten",
            "Falsche Prognosen möglich",
            "Integration in MES/WMS komplex"
        ],
        "Gegenmaßnahme": [
            "Kombination mit Vibration und Spindelstrom",
            "Pilotphase mit echten CNC-Daten",
            "Confidence Score + Sicherheitsmarge",
            "Schnittstellen über OPC-UA, MQTT, REST API"
        ]
    })

    st.dataframe(risks, use_container_width=True, hide_index=True)

    st.subheader("Technologie-Stack")

    tech = pd.DataFrame({
        "Technologie": [
            "Python + Streamlit",
            "Scikit-Learn",
            "Librosa",
            "Plotly",
            "Google Gemini AI",
            "Pandas + NumPy",
            "SoundFile"
        ],
        "Verwendung": [
            "Dashboard und Web-Interface",
            "Random Forest KI-Modell",
            "Audio Feature Extraction",
            "Interaktive Visualisierungen",
            "KI-Chat Assistant",
            "Datenverarbeitung und Simulation",
            "Audio-Generierung und -Verarbeitung"
        ]
    })

    st.dataframe(tech, use_container_width=True, hide_index=True)
# =========================================================
# Tab 6: KPI Simulation
# =========================================================

with tab6:
    st.header("📊 KPI Simulation: Traditionell vs. Predictive Tool Logistics")

    kpi_events = st.slider("Anzahl simulierter Werkzeugereignisse", 50, 500, 180, 10)

    kpi_df, summary = simulate_kpis(n_events=kpi_events, seed=int(seed) + 400)

    st.subheader("Zusammenfassung")

    st.dataframe(summary, use_container_width=True)

    traditional = summary[summary["Methode"] == "Traditionell"].iloc[0]
    predictive = summary[summary["Methode"] == "Predictive Tool Logistics"].iloc[0]

    downtime_reduction = (
        1 - predictive["Gesamtstillstand_min"] / traditional["Gesamtstillstand_min"]
    ) * 100

    emergency_reduction = (
        1 - predictive["Eiltransporte"] / max(traditional["Eiltransporte"], 1)
    ) * 100

    cost_reduction = (
        1 - predictive["Stillstandskosten_EUR"] / traditional["Stillstandskosten_EUR"]
    ) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Stillstandsreduktion", f"{downtime_reduction:.1f}%", "gegenüber traditionell", "#22c55e")
    with col2:
        kpi_card("Eiltransport-Reduktion", f"{emergency_reduction:.1f}%", "weniger Störungen", "#38bdf8")
    with col3:
        kpi_card("Kostenreduktion", f"{cost_reduction:.1f}%", "simulierte Stillstandskosten", "#a855f7")

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.bar(
            summary,
            x="Methode",
            y="Gesamtstillstand_min",
            color="Methode",
            title="Gesamter Maschinenstillstand [min]"
        )
        fig1.update_layout(paper_bgcolor="#111827", plot_bgcolor="#0f172a", font=dict(color="white"))
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.bar(
            summary,
            x="Methode",
            y="Eiltransporte",
            color="Methode",
            title="Anzahl Eiltransporte"
        )
        fig2.update_layout(paper_bgcolor="#111827", plot_bgcolor="#0f172a", font=dict(color="white"))
        st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)

    with colC:
        fig3 = px.bar(
            summary,
            x="Methode",
            y="Rechtzeitige_Bereitstellung",
            color="Methode",
            title="Anteil rechtzeitiger Bereitstellungen"
        )
        fig3.update_layout(paper_bgcolor="#111827", plot_bgcolor="#0f172a", font=dict(color="white"))
        st.plotly_chart(fig3, use_container_width=True)

    with colD:
        fig4 = px.bar(
            summary,
            x="Methode",
            y="Durchschnittliches_Ausschussrisiko",
            color="Methode",
            title="Durchschnittliches Ausschussrisiko"
        )
        fig4.update_layout(paper_bgcolor="#111827", plot_bgcolor="#0f172a", font=dict(color="white"))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    Der Predictive-Tool-Logistics-Ansatz reduziert Stillstand, weil Werkzeugverschleiß früher erkannt und direkt in logistische Bereitstellungsaufträge umgesetzt wird.
    """)
