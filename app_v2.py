import streamlit as st
import streamlit as st
import streamlit as st

st.set_page_config(layout="wide")

st.title("Predictive Logistics System")

# حفظ الآلات
if "machines" not in st.session_state:
    st.session_state.machines = []

# إضافة آلة جديدة
st.sidebar.header("Neue Maschine hinzufügen")

machine_name = st.sidebar.text_input("Maschinenname")

sicherheitsmarge = st.sidebar.slider(
    "Sicherheitsmarge [min]",
    0, 20, 5
)

wahrscheinlichkeit = st.sidebar.slider(
    "Wahrscheinlichkeit Werkzeug nicht auf Lager",
    0.0, 1.0, 0.20
)

# زر الإضافة
if st.sidebar.button("Maschine hinzufügen"):

    st.session_state.machines.append({
        "name": machine_name,
        "sicherheitsmarge": sicherheitsmarge,
        "wahrscheinlichkeit": wahrscheinlichkeit
    })

# عرض جميع الآلات
for machine in st.session_state.machines:

    st.header(machine["name"])

    st.write(
        "Sicherheitsmarge:",
        machine["sicherheitsmarge"]
    )

    st.write(
        "Wahrscheinlichkeit:",
        machine["wahrscheinlichkeit"]
    )

    st.progress(machine["wahrscheinlichkeit"])

    st.markdown("---")

st.set_page_config(layout="wide")

st.markdown("""
<style>

/* لون الصفحة */
.main {
    background-color: white;
}

/* جميع النصوص */
html, body, [class*="css"] {
    color: black !important;
    font-size: 18px !important;
}

/* تكبير الكتابة الجانبية */
section[data-testid="stSidebar"] {
    width: 380px !important;
    background-color: #f5f5f5;
}

/* عناوين السلايدر */
.stSlider label {
    color: black !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

/* النصوص داخل السايدبار */
section[data-testid="stSidebar"] * {
    color: black !important;
}

/* العناوين */
h1, h2, h3 {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
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
# Page Config
# =========================================================

st.set_page_config(
    page_title="Predictive Tool Logistics – Advanced Prototype",
    page_icon="🏭",
    layout="wide"
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

st.markdown("""
<div class="hero">
    <div class="hero-title">🏭 Predictive Tool Logistics Control Tower</div>
    <div class="hero-subtitle">
        KI-gestützte akustische Werkzeugzustandserkennung mit automatischer Werkzeugbereitstellung in der CNC-Fertigung
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Sidebar Controls
# =========================================================

st.sidebar.header("⚙️ Prototype-Konfiguration")

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
    "🏭 Control Tower",
    "🔍 Maschinen-Detail",
    "🎧 Audio & KI Analyse",
    "🚚 Logistik-Leitstand",
    "📊 KPI Simulation",
    "🧩 Architektur & Pitch"
])


# =========================================================
# Tab 1: Control Tower
# =========================================================

with tab1:
    st.header("🏭 Smart Factory Control Tower")

    st.markdown("""
    <div class="section-card">
    Diese Ansicht zeigt den aktuellen Zustand der CNC-Fertigung. 
    Die Größe der Punkte in der Fabrikkarte entspricht dem Risiko. 
    Die Farbe entspricht der logistischen Entscheidung.
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(factory_map(fleet), use_container_width=True)

    st.subheader("Priorisierte Maschinenliste")

    display_cols = [
        "Maschine", "Zelle", "Maschinentyp", "Werkzeug_ID", "Werkzeugtyp",
        "Material", "KI_Zustand", "Confidence", "RUL_min",
        "Logistische_Vorlaufzeit_min", "Entscheidung", "Risk_Score"
    ]

    st.dataframe(
        fleet[display_cols],
        use_container_width=True,
        height=420
    )

    st.subheader("Entscheidungslegende")

    legend_cols = st.columns(4)
    decisions = ["MONITORING", "VORWARNUNG", "AUTO_AUFTRAG", "BEDIENER_FREIGABE",
                 "UNSICHER_WARNUNG", "SOFORT_STOPP", "BESTANDSRISIKO"]

    for i, d in enumerate(decisions):
        with legend_cols[i % 4]:
            st.markdown(badge(d, DECISION_COLORS[d]), unsafe_allow_html=True)


# =========================================================
# Tab 2: Machine Detail
# =========================================================

with tab2:
    st.header("🔍 Maschinen-Detailansicht")

    selected_machine = st.selectbox(
        "Maschine auswählen",
        fleet["Maschine"].tolist(),
        index=0
    )

    selected = fleet[fleet["Maschine"] == selected_machine].iloc[0]

    colA, colB, colC, colD = st.columns(4)

    with colA:
        kpi_card("Maschine", selected["Maschine"], selected["Maschinentyp"], "#38bdf8")
    with colB:
        kpi_card("Werkzeug", selected["Werkzeug_ID"], selected["Werkzeugtyp"], "#22c55e")
    with colC:
        kpi_card("KI-Zustand", selected["KI_Zustand"], f"Confidence {selected['Confidence']*100:.1f}%", STATE_COLORS[selected["KI_Zustand"]])
    with colD:
        kpi_card("Entscheidung", selected["Entscheidung"], f"Risk Score {selected['Risk_Score']}", DECISION_COLORS[selected["Entscheidung"]])

    st.markdown("---")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Produktionsauftrag")

        info_df = pd.DataFrame({
            "Parameter": [
                "Zelle",
                "Auftrag",
                "Material",
                "RPM",
                "Zähne",
                "Restteile",
                "Fällig in",
                "Stillstandskosten"
            ],
            "Wert": [
                selected["Zelle"],
                selected["Auftrag"],
                selected["Material"],
                selected["RPM"],
                selected["Zähne"],
                selected["Restteile"],
                f"{selected['Fällig_in_min']} min",
                f"{selected['Stillstandskosten_EUR_min']} €/min"
            ]
        })

        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Werkzeuglogistische Bewertung")

        logistics_df = pd.DataFrame({
            "Kennzahl": [
                "RUL",
                "Logistische Vorlaufzeit",
                "Lagerentnahme",
                "Voreinstellung",
                "AGV-Wartezeit",
                "Transport",
                "Sicherheitsmarge",
                "Bestandsverzug",
                "Bestand OK"
            ],
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

    decision = selected["Entscheidung"]

    if decision == "AUTO_AUFTRAG":
        st.success("Automatischer Werkzeugbereitstellungsauftrag wurde erzeugt.")
    elif decision == "SOFORT_STOPP":
        st.error("Sofortiger Stopp empfohlen: Werkzeugzustand Replace und RUL kleiner als Vorlaufzeit.")
    elif decision == "BEDIENER_FREIGABE":
        st.warning("Werkzeugbereitstellung empfohlen, aber Bedienerfreigabe erforderlich.")
    elif decision == "BESTANDSRISIKO":
        st.error("Bestandsrisiko: Ersatzwerkzeug ist nicht direkt verfügbar. Sonderbeschaffung oder Priorisierung notwendig.")
    elif decision == "VORWARNUNG":
        st.info("Vorwarnung: Noch kein Auftrag notwendig, aber logistische Vorbereitung empfohlen.")
    else:
        st.info("Monitoring: Werkzeug kann weiterlaufen.")


# =========================================================
# Tab 3: Audio and AI
# =========================================================

with tab3:
    st.header("🎧 Akustische Werkzeuganalyse und KI-Klassifikation")

    selected_audio_machine = st.selectbox(
        "Maschine für Audioanalyse auswählen",
        fleet["Maschine"].tolist(),
        index=0,
        key="audio_machine_select"
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

    st.subheader(f"Audiosignal von {audio_row['Maschine']} – Werkzeug {audio_row['Werkzeug_ID']}")

    st.audio(audio_to_wav_bytes(audio), format="audio/wav")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_waveform(audio))
        st.pyplot(plot_spectrum(audio))

    with col2:
        st.pyplot(plot_mel(audio))

        features = extract_features(audio).reshape(1, -1)
        probas = model.predict_proba(features)[0]
        classes = model.classes_

        proba_df = pd.DataFrame({
            "Zustand": classes,
            "Wahrscheinlichkeit": probas
        })

        fig_prob = px.bar(
            proba_df,
            x="Zustand",
            y="Wahrscheinlichkeit",
            color="Zustand",
            color_discrete_map=STATE_COLORS,
            title="KI-Wahrscheinlichkeiten pro Werkzeugzustand",
            text="Wahrscheinlichkeit"
        )

        fig_prob.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_prob.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            font=dict(color="white"),
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Tatsächlicher Zustand", audio_row["Ist_Zustand"], "simuliert", STATE_COLORS[audio_row["Ist_Zustand"]])
    with c2:
        kpi_card("KI-Zustand", audio_row["KI_Zustand"], "Vorhersage", STATE_COLORS[audio_row["KI_Zustand"]])
    with c3:
        kpi_card("Confidence", f"{audio_row['Confidence']*100:.1f}%", "Modellsicherheit", "#38bdf8")
    with c4:
        kpi_card("Test Accuracy", f"{model_accuracy*100:.1f}%", "synthetische Testdaten", "#a855f7")

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    st.dataframe(cm_df, use_container_width=True)


# =========================================================
# Tab 4: Logistics Control
# =========================================================

with tab4:
    st.header("🚚 Logistik-Leitstand: Werkzeuglager, Voreinstellung und AGV/FTS")

    order_filter = fleet[fleet["Entscheidung"].isin([
        "SOFORT_STOPP",
        "BESTANDSRISIKO",
        "AUTO_AUFTRAG",
        "BEDIENER_FREIGABE",
        "UNSICHER_WARNUNG",
        "VORWARNUNG"
    ])].copy()

    if order_filter.empty:
        st.info("Aktuell keine Werkzeugbereitstellungsaufträge notwendig.")
    else:
        st.subheader("Order Board – priorisierte Werkzeugbereitstellung")

        order_board = order_filter[[
            "Maschine", "Zelle", "Werkzeug_ID", "Werkzeugtyp", "KI_Zustand",
            "RUL_min", "Logistische_Vorlaufzeit_min", "Entscheidung",
            "Bestand_OK", "Risk_Score"
        ]]

        st.dataframe(order_board, use_container_width=True, height=350)

    st.markdown("---")

    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("Ressourcenstatus: AGV / FTS")

        critical_orders = order_filter.head(4)

        agv_rows = []
        for i in range(1, 5):
            if i <= len(critical_orders):
                r = critical_orders.iloc[i - 1]
                status = f"Transportauftrag für {r['Maschine']}"
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

        st.dataframe(pd.DataFrame(agv_rows), use_container_width=True, hide_index=True)

    with colR:
        st.subheader("Ressourcenstatus: Voreinstellstationen")

        preset_rows = []
        for i in range(1, 4):
            if i <= len(order_filter):
                r = order_filter.iloc[i - 1]
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

        st.dataframe(pd.DataFrame(preset_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    if not order_filter.empty:
        timeline_machine = st.selectbox(
            "Timeline für Auftrag anzeigen",
            order_filter["Maschine"].tolist()
        )

        timeline_row = order_filter[order_filter["Maschine"] == timeline_machine].iloc[0]

        st.plotly_chart(create_timeline(timeline_row), use_container_width=True)

        st.subheader("Automatisch erzeugte Logistikaktionen")

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
                "KI-Modul",
                "Werkzeuglager",
                "WMS / Tool Management",
                "Voreinstellstation",
                "FTS-Leitsystem",
                "AGV / Mitarbeiter",
                "MES / Bedienerinterface"
            ],
            "Status": [
                "Abgeschlossen",
                "Abgeschlossen" if timeline_row["Bestand_OK"] else "Problem",
                "Gestartet",
                "Geplant",
                "Geplant",
                "In Vorbereitung",
                "Offen"
            ]
        })

        st.dataframe(actions, use_container_width=True, hide_index=True)


# =========================================================
# Tab 5: KPI Simulation
# =========================================================

with tab5:
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


# =========================================================
# Tab 6: Architecture and Pitch
# =========================================================

with tab6:
    st.header("🧩 Architektur, Innovationskern und Präsentations-Pitch")

    st.subheader("Systemarchitektur")

    st.code("""
Akustische Werkzeugsignale
        ↓
Audio Feature Extraction
        ↓
KI-Modell zur Werkzeugzustandserkennung
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
    """, language="text")

    st.subheader("Innovationskern")

    st.success("""
    Der Prototyp ist kein reines Monitoring-Dashboard. 
    Er wandelt akustische Sensordaten in eine konkrete produktionslogistische Entscheidung um:
    Wann muss ein Ersatzwerkzeug vorbereitet und zur Maschine transportiert werden?
    """)

    st.subheader("Industrielles Problem")

    st.markdown("""
    In CNC-Fertigungen entstehen häufig Maschinenstillstände, weil Werkzeugverschleiß zwar erkannt wird, 
    die Werkzeuglogistik aber zu spät reagiert. 
    Das Ersatzwerkzeug muss nicht nur vorhanden sein, sondern auch rechtzeitig reserviert, 
    voreingestellt und zur richtigen Maschine transportiert werden.
    """)

    st.subheader("Präsentations-Pitch auf Deutsch")

    st.info("""
    Unser Projekt „Predictive Tool Logistics“ verbindet KI-basierte akustische Werkzeugzustandserkennung 
    mit automatisierter Produktionslogistik. 
    Der Prototyp analysiert simulierte Werkzeugsignale von CNC-Maschinen, prognostiziert den Werkzeugzustand 
    und die Restlebensdauer und vergleicht diese mit der logistischen Vorlaufzeit für Lagerentnahme, 
    Werkzeugvoreinstellung und Transport. 
    Wenn die Restlebensdauer nicht mehr ausreicht, wird automatisch ein Bereitstellungsauftrag erzeugt. 
    Dadurch werden ungeplante Maschinenstillstände, Eiltransporte und Ausschussrisiken reduziert.
    """)

    st.subheader("Warum ist das Industrie 4.0?")

    ind40 = pd.DataFrame({
        "Industrie-4.0-Baustein": [
            "IoT / Sensorik",
            "KI / Machine Learning",
            "Predictive Analytics",
            "Produktionslogistik",
            "Automatisierung",
            "Digital Twin / Simulation",
            "Smart Factory"
        ],
        "Umsetzung im Prototyp": [
            "Akustische Werkzeugsignale",
            "Klassifikation von Werkzeugzuständen",
            "RUL-Prognose",
            "Werkzeuglager, Voreinstellung, AGV/FTS",
            "Automatische Bereitstellungsentscheidung",
            "Simulierte CNC-Fabrik mit vielen Maschinen",
            "Control Tower mit Echtzeitlogik"
        ]
    })

    st.dataframe(ind40, use_container_width=True, hide_index=True)

    st.subheader("Risiken und Gegenmaßnahmen")

    risks = pd.DataFrame({
        "Risiko": [
            "Reale Fabrikgeräusche können die Audioanalyse stören",
            "KI braucht echte Trainingsdaten",
            "Falsche Prognosen können zu frühen oder späten Wechseln führen",
            "Integration in reale MES/WMS/FTS-Systeme ist komplex"
        ],
        "Gegenmaßnahme": [
            "Kombination mit Vibration, Spindelstrom oder Drehmoment",
            "Pilotphase mit echten CNC-Daten und Transfer Learning",
            "Confidence Score, Sicherheitsmarge und Bedienerfreigabe",
            "Schnittstellen über MQTT, OPC UA, REST API oder Node-RED"
        ]
    })

    st.dataframe(risks, use_container_width=True, hide_index=True)
