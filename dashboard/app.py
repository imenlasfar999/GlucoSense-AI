import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import requests
import time
import tempfile
from html import escape
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from fpdf import FPDF

DATA_FILE      = "latest_sensor_data.csv"
HISTORY_FILE   = "live_prediction_history.csv"
PATIENTS_FILE  = "patients_history.json"
GEMINI_MODEL   = "gemini-2.5-flash"

model    = joblib.load("xgboost_glucose_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="GlucoSense AI", page_icon="🩸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --ink:#0f172a;
  --muted:#64748b;
  --glass:rgba(255,255,255,.72);
  --glass-strong:rgba(255,255,255,.88);
  --stroke:rgba(148,163,184,.24);
  --pink:#ec4899;
  --violet:#8b5cf6;
  --blue:#0ea5e9;
  --cyan:#06b6d4;
  --green:#22c55e;
  --amber:#f59e0b;
  --red:#ef4444;
}

*{font-family:'Inter',sans-serif;}
html, body, [class*="css"]{scroll-behavior:smooth;}

.stApp{
  color:var(--ink);
  background:
    radial-gradient(circle at 10% 10%, rgba(236,72,153,.20), transparent 27%),
    radial-gradient(circle at 86% 12%, rgba(14,165,233,.22), transparent 30%),
    radial-gradient(circle at 45% 92%, rgba(139,92,246,.20), transparent 33%),
    linear-gradient(135deg,#fff7fb 0%,#f4f0ff 35%,#eaf7ff 72%,#edfff8 100%);
  overflow-x:hidden;
}
.stApp::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.42) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.42) 1px, transparent 1px);
  background-size:44px 44px;
  mask-image:linear-gradient(to bottom, rgba(0,0,0,.34), transparent 75%);
}
.stApp::after{
  content:"";position:fixed;width:520px;height:520px;right:-180px;top:170px;pointer-events:none;z-index:0;
  background:radial-gradient(circle,rgba(34,211,238,.22),rgba(236,72,153,.12),transparent 70%);
  filter:blur(10px);
}

[data-testid="stAppViewContainer"]>.main{position:relative;z-index:1;}
.main .block-container{max-width:1480px;padding-top:1.2rem;padding-bottom:3rem;}

/* Sidebar - premium control panel */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(255,255,255,.86),rgba(248,250,252,.72)) !important;
  backdrop-filter:blur(28px);
  border-right:1px solid rgba(148,163,184,.24);
  box-shadow:18px 0 60px rgba(99,102,241,.13);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
  color:#0f172a;font-weight:900;letter-spacing:-.02em;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#334155;}
[data-testid="stSidebar"] .stButton>button,
[data-testid="stSidebar"] .stDownloadButton>button{
  width:100%;border-radius:16px;border:1px solid rgba(148,163,184,.22);
  background:linear-gradient(135deg,#ffffff,#f8fafc);color:#0f172a;font-weight:800;
  box-shadow:0 8px 22px rgba(15,23,42,.08);transition:all .22s ease;
}
[data-testid="stSidebar"] .stButton>button:hover{
  transform:translateY(-2px);border-color:rgba(139,92,246,.42);box-shadow:0 14px 34px rgba(139,92,246,.16);
}

/* Inputs */
div[data-baseweb="select"]>div,
div[data-baseweb="input"]>div,
textarea,
input{
  border-radius:15px!important;border:1px solid rgba(148,163,184,.28)!important;
  background:rgba(255,255,255,.78)!important;box-shadow:0 8px 25px rgba(15,23,42,.06)!important;
}
label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stTextArea label{
  font-weight:800!important;color:#334155!important;letter-spacing:-.01em;
}

/* Premium hero */
.hero{
  position:relative;overflow:hidden;color:white;border-radius:38px;padding:42px 44px;margin:4px 0 32px;
  background:
    radial-gradient(circle at 18% 18%,rgba(255,255,255,.30),transparent 18%),
    radial-gradient(circle at 78% 15%,rgba(34,211,238,.46),transparent 28%),
    radial-gradient(circle at 68% 86%,rgba(34,197,94,.28),transparent 24%),
    linear-gradient(135deg,#db2777 0%,#8b5cf6 38%,#2563eb 68%,#06b6d4 100%);
  box-shadow:0 24px 75px rgba(59,130,246,.31),0 14px 38px rgba(236,72,153,.18);
  border:1px solid rgba(255,255,255,.30);
}
.hero:before{
  content:"";position:absolute;inset:-45%;opacity:.45;
  background:conic-gradient(from 90deg,transparent,rgba(255,255,255,.45),transparent,rgba(34,211,238,.32),transparent);
  animation:heroSpin 16s linear infinite;
}
.hero:after{
  content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(255,255,255,.12),transparent 38%,rgba(255,255,255,.08));
  pointer-events:none;
}
@keyframes heroSpin{to{transform:rotate(360deg)}}
.hero-content{position:relative;z-index:2;display:grid;grid-template-columns:1.55fr .85fr;gap:24px;align-items:center;}
.hero-kicker{display:inline-flex;gap:8px;align-items:center;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.28);padding:9px 14px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:2px;text-transform:uppercase;backdrop-filter:blur(10px);}
.hero h1{font-size:58px;line-height:1.02;margin:18px 0 10px;font-weight:950;letter-spacing:-.045em;text-shadow:0 10px 28px rgba(15,23,42,.20);}
.hero h2{font-size:27px;margin:0 0 16px;font-weight:800;letter-spacing:-.02em;color:rgba(255,255,255,.96);}
.hero p{font-size:16px;color:rgba(255,255,255,.88);margin:6px 0;}
.hero-flow{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px;}
.flow-chip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.26);padding:10px 13px;border-radius:999px;font-weight:900;font-size:12px;backdrop-filter:blur(12px);box-shadow:inset 0 1px 0 rgba(255,255,255,.18);}
.hero-panel{background:rgba(255,255,255,.17);border:1px solid rgba(255,255,255,.25);border-radius:30px;padding:24px;backdrop-filter:blur(18px);box-shadow:inset 0 1px 0 rgba(255,255,255,.20),0 18px 50px rgba(15,23,42,.18);}
.hero-panel-title{font-size:13px;text-transform:uppercase;letter-spacing:2px;font-weight:900;color:rgba(255,255,255,.72);}
.hero-panel-value{font-size:36px;line-height:1;font-weight:950;margin:10px 0;}
.hero-mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px;}
.hero-mini{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:12px;font-size:12px;font-weight:800;}
.pulse-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:8px;box-shadow:0 0 0 8px rgba(34,197,94,.16);animation:pulse 1.6s infinite;}
@keyframes pulse{0%{transform:scale(.95)}50%{transform:scale(1.12)}100%{transform:scale(.95)}}

/* Cards and values */
.card,.glucose-card,.status-box{
  position:relative;overflow:hidden;border:1px solid rgba(255,255,255,.62);
  box-shadow:0 22px 58px rgba(15,23,42,.13), inset 0 1px 0 rgba(255,255,255,.58);
  backdrop-filter:blur(18px);
}
.card{
  background:linear-gradient(145deg,rgba(255,255,255,.92),rgba(255,255,255,.68));
  padding:26px;border-radius:30px;margin-bottom:18px;transition:all .25s ease;
}
.card:hover,.glucose-card:hover,.status-box:hover{transform:translateY(-4px);box-shadow:0 30px 70px rgba(99,102,241,.17), inset 0 1px 0 rgba(255,255,255,.64);}
.card:before,.glucose-card:before,.status-box:before{content:"";position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--pink),var(--violet),var(--cyan),var(--green));}
.glucose-card{background:linear-gradient(135deg,#0ea5e9,#6366f1 58%,#a855f7);color:white;padding:42px;border-radius:34px;text-align:center;}
.glucose-card:after{content:"";position:absolute;width:220px;height:220px;border-radius:50%;right:-70px;top:-85px;background:rgba(255,255,255,.18);}
.glucose-value{font-size:86px;font-weight:950;line-height:.95;letter-spacing:-.06em;text-shadow:0 10px 28px rgba(15,23,42,.22);}
.unit{font-size:24px;font-weight:900;color:rgba(255,255,255,.85);}
.status-box{padding:40px 30px;border-radius:34px;text-align:center;font-size:38px;font-weight:950;color:white;min-height:154px;display:flex;align-items:center;justify-content:center;letter-spacing:-.03em;}
.normal{background:linear-gradient(135deg,#059669,#22c55e,#86efac);}
.pre{background:linear-gradient(135deg,#f59e0b,#f97316,#fb7185);}
.high{background:linear-gradient(135deg,#dc2626,#ef4444,#fb7185);}
.low{background:linear-gradient(135deg,#2563eb,#38bdf8,#67e8f9);}
.sensor-title{font-size:22px;font-weight:950;color:#0f172a;letter-spacing:-.03em;}
.value{font-size:17px;color:#475569;font-weight:700;}
.live-badge{display:inline-flex;align-items:center;gap:8px;padding:12px 18px;border-radius:999px;background:linear-gradient(135deg,#dcfce7,#ecfeff);color:#065f46;font-weight:950;border:1px solid rgba(34,197,94,.24);box-shadow:0 14px 36px rgba(34,197,94,.13);margin:8px 0 20px;}

/* Streamlit sections */
.block-container h2{
  display:inline-flex;align-items:center;gap:10px;margin-top:28px;margin-bottom:12px;
  font-size:28px!important;font-weight:950!important;letter-spacing:-.035em;color:#0f172a!important;
  padding:10px 16px;border-radius:18px;background:rgba(255,255,255,.62);border:1px solid rgba(255,255,255,.66);box-shadow:0 14px 32px rgba(15,23,42,.08);
}
.block-container h3{font-weight:900!important;color:#1e293b!important;}
.stProgress > div > div > div > div{background:linear-gradient(90deg,#60a5fa,#22c55e,#f59e0b,#ef4444)!important;border-radius:999px;}
.stProgress > div > div{height:14px!important;background:rgba(255,255,255,.65)!important;border-radius:999px;border:1px solid rgba(148,163,184,.20);}

/* Buttons */
.stButton>button,.stDownloadButton>button{
  border:none!important;border-radius:18px!important;font-weight:950!important;letter-spacing:-.01em!important;
  background:linear-gradient(135deg,#ec4899,#8b5cf6,#0ea5e9)!important;color:white!important;
  box-shadow:0 16px 40px rgba(139,92,246,.24)!important;transition:all .22s ease!important;
}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-3px)!important;box-shadow:0 24px 54px rgba(139,92,246,.32)!important;filter:saturate(1.1);}

/* Dataframes */
[data-testid="stDataFrame"]{border-radius:24px!important;overflow:hidden;border:1px solid rgba(148,163,184,.22);box-shadow:0 20px 50px rgba(15,23,42,.10);}

.footer{background:linear-gradient(135deg,rgba(255,255,255,.85),rgba(255,255,255,.58));padding:24px;border-radius:28px;color:#334155;border:1px solid rgba(255,255,255,.65);box-shadow:0 18px 48px rgba(15,23,42,.10);backdrop-filter:blur(18px);}

@media(max-width:1000px){.hero-content{grid-template-columns:1fr}.hero h1{font-size:42px}.cards-grid{grid-template-columns:1fr 1fr}.glucose-value{font-size:64px}}
</style>
""", unsafe_allow_html=True)

def clarke_zone(real, pred):
    real,pred=float(real),float(pred)
    if (real<70 and pred<70) or abs(pred-real)<=0.20*real: return "A"
    if (real<=70 and pred>=180) or (real>=180 and pred<=70): return "E"
    if 70<=real<=290 and pred>=real+110: return "C"
    if 130<=real<=180 and pred<=(7/5)*real-182: return "C"
    if real>=240 and 70<=pred<=180: return "D"
    if real<=70 and 70<=pred<=180: return "D"
    return "B"

def load_history():
    if os.path.exists(HISTORY_FILE): return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Measurement_ID","Timestamp","State","Time_After_Meal_min","Predicted_Glucose","Status","Reference_Glucose","Error","Absolute_Error","Clarke_Zone"])

def save_history(df): df.to_csv(HISTORY_FILE,index=False)

def load_patients():
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE) as f: return json.load(f)
    return {}

def save_patients(d):
    with open(PATIENTS_FILE,"w") as f: json.dump(d,f,indent=2)

def calculate_bmi(weight_kg, height_cm):
    try:
        height_m = float(height_cm) / 100
        if height_m <= 0:
            return None
        return float(weight_kg) / (height_m ** 2)
    except Exception:
        return None

def classify_bmi(bmi):
    if bmi is None:
        return "Not available"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"       # ← FIXED: capital N (WHO standard)
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity class I"
    elif bmi < 40:
        return "Obesity class II"
    else:
        return "Obesity class III"

def get_gemini_api_key():
    return "YOUR_API_KEY_HERE"

def call_gemini(prompt):
    api_key = get_gemini_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents":[{"parts":[{"text":prompt}]}],
        "generationConfig":{"temperature":0.75,"topP":0.95,"maxOutputTokens":5000}  # ← FIXED: was 3500
    }
    for attempt in range(4):
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"], None
            elif r.status_code in (503,429):
                time.sleep((attempt+1)*8)
                continue
            else:
                return None, f"API ERROR {r.status_code}: {r.text}"
        except Exception as e:
            return None, f"ERROR: {str(e)}"
    return None, "Server busy. Please try again in 1 minute."

def build_premium_prompt(patient, prediction, status, state_label, time_after_meal, history_list, real_glucose, abs_error, zone, accuracy):
    if history_list:
        history_text = "PREVIOUS VISITS (most recent first):\n"
        for h in reversed(history_list[-5:]):
            history_text += f"  - {h['date']}: glucose {h['glucose']} mg/dL | status: {h['status']} | state: {h['state']}\n"
    else:
        history_text = "No previous visits. This is the FIRST visit — establish as baseline."

    val_text = "No glucometer reference provided for this session."
    if real_glucose > 0:
        val_text = f"Reference glucometer: {real_glucose:.1f} mg/dL | Absolute error: {abs_error:.1f} mg/dL | Approximate agreement score: {accuracy:.1f}% | Clarke Zone: {zone}"

    return f"""You are GlucoSense AI Agent — an advanced clinical research AI integrated into a non-invasive glucose monitoring system.

Generate a COMPREHENSIVE, DETAILED, IMPRESSIVE clinical research report. Minimum 800 words total.
Each section must have at least 4-6 full sentences with real scientific depth.
Write like a senior medical researcher. Be specific, analytical, and professional.
Use the patient name and specific numbers throughout.
Important safety style: avoid diagnostic claims, avoid treatment prescriptions, and use research-oriented language such as "suggests", "is consistent with", and "requires clinical confirmation".

PATIENT: {patient['name']} | Code: {patient['code']} | Age: {patient['age']} yrs | Gender: {patient['gender']} | Weight: {patient.get('weight','?')} kg | Height: {patient.get('height','?')} cm | BMI: {patient.get('bmi','Not available')} ({patient.get('bmi_category','Not available')}) | Diabetes: {patient.get('diabetes','?')} | Family history: {patient.get('family_history','?')} | Physical activity: {patient.get('physical_activity','?')}
MEASUREMENT: {state_label} | {time_after_meal} min post-meal | Predicted glucose: {prediction:.1f} mg/dL | Status: {status}
VALIDATION: {val_text}
HISTORY: {history_text}

Write the report in Markdown with EXACTLY these 8 section headers (use ## for each):

## Executive Summary
4-6 sentences. Summarize the entire session for {patient['name']}. Mention the glucose value {prediction:.1f} mg/dL, the status {status}, measurement context, and clinical relevance of this result.

## Patient-Specific Clinical Context
4-6 sentences. Analyze how {patient['name']}'s profile (age {patient['age']}, gender {patient['gender']}, weight {patient.get('weight','?')} kg, height {patient.get('height','?')} cm, BMI {patient.get('bmi','Not available')} classified as {patient.get('bmi_category','Not available')}, diabetes history: {patient.get('diabetes','?')}, family history: {patient.get('family_history','?')}, and physical activity: {patient.get('physical_activity','?')}) influences the interpretation of this glucose reading. Discuss physiological context specific to this profile.

## Glucose Prediction Analysis
4-6 sentences. Analyze the predicted value {prediction:.1f} mg/dL in depth. Explain the clinical thresholds used, why this value falls in the {status} category, and what this means physiologically for a {state_label} measurement at {time_after_meal} minutes post-meal.

## Validation and Agreement Assessment
4-6 sentences. If reference glucometer was provided, analyze the absolute error, approximate agreement score, and Clarke zone in detail. Explain that the agreement score is a dashboard indicator, not a formal machine-learning accuracy metric. If not provided, explain what the Clarke Error Grid is, why validation matters for research prototypes, and what future validation would look like.

## Longitudinal Trend Analysis
4-6 sentences. Compare this visit with previous visits if available. Describe the health trajectory. If first visit, explain baseline establishment and what to monitor in future sessions.

## Personalized Monitoring Guidance
5 specific numbered recommendations tailored exactly to {patient['name']}'s profile, BMI category, family history, physical activity level, glucose result, and measurement context. Be specific and actionable, but do not prescribe treatment.

## System Technical Limitations
4-5 sentences. Discuss limitations of MQ138/MQ3/MQ6 VOC sensors, MAX30102 PPG signals, environmental interference from DHT22 data, the 120-sample training dataset size, and prototype research status.

## Research and Clinical Disclaimer
3-4 sentences. State clearly this is for academic research and educational demonstration only. Mention that certified glucometer confirmation is always required for medical decisions.
"""

def simple_markdown_to_html(md_text):
    html_lines = []
    in_ul = False
    in_ol = False
    ol_counter = 0
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            if in_ul: html_lines.append("</ul>"); in_ul=False
            if in_ol: html_lines.append("</ol>"); in_ol=False; ol_counter=0
            continue
        if line.startswith("## "):
            if in_ul: html_lines.append("</ul>"); in_ul=False
            if in_ol: html_lines.append("</ol>"); in_ol=False; ol_counter=0
            html_lines.append(f"<h2>{escape(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{escape(line[4:].strip())}</h3>")
        elif line.startswith("**") and line.endswith("**"):
            html_lines.append(f"<p><strong>{escape(line[2:-2])}</strong></p>")
        elif line.startswith("- ") or line.startswith("• "):
            if in_ol: html_lines.append("</ol>"); in_ol=False; ol_counter=0
            if not in_ul: html_lines.append("<ul>"); in_ul=True
            item = line[2:].strip() if line.startswith("- ") else line[2:].strip()
            import re
            item_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escape(item))
            html_lines.append(f"<li>{item_html}</li>")
        elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            if in_ul: html_lines.append("</ul>"); in_ul=False
            if not in_ol: html_lines.append("<ol>"); in_ol=True
            rest = line[2:].strip() if len(line)>2 else line
            import re
            item_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escape(rest))
            html_lines.append(f"<li>{item_html}</li>")
        else:
            if in_ul: html_lines.append("</ul>"); in_ul=False
            if in_ol: html_lines.append("</ol>"); in_ol=False; ol_counter=0
            import re
            p_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escape(line))
            html_lines.append(f"<p>{p_html}</p>")
    if in_ul: html_lines.append("</ul>")
    if in_ol: html_lines.append("</ol>")
    return "\n".join(html_lines)

def enhance_ai_report_html(report_html):
    replacements = {
        "<h2>Executive Summary</h2>": "<h2 class=\"sec-summary\"><span>🧠</span> Executive Summary</h2>",
        "<h2>Patient-Specific Clinical Context</h2>": "<h2 class=\"sec-patient\"><span>👤</span> Patient Profile & Clinical Context</h2>",
        "<h2>Glucose Prediction Analysis</h2>": "<h2 class=\"sec-glucose\"><span>🩸</span> Glucose Assessment</h2>",
        "<h2>Validation and Agreement Assessment</h2>": "<h2 class=\"sec-validation\"><span>📊</span> Validation Assessment</h2>",
        "<h2>Longitudinal Trend Analysis</h2>": "<h2 class=\"sec-trend\"><span>📈</span> Trend Analysis</h2>",
        "<h2>Personalized Monitoring Guidance</h2>": "<h2 class=\"sec-guidance\"><span>💡</span> Personalized Guidance</h2>",
        "<h2>System Technical Limitations</h2>": "<h2 class=\"sec-limits\"><span>⚙️</span> Technical Limitations</h2>",
        "<h2>Research and Clinical Disclaimer</h2>": "<h2 class=\"sec-disclaimer\"><span>⚠️</span> Research Disclaimer</h2>",
    }
    for old, new in replacements.items():
        report_html = report_html.replace(old, new)
    return report_html



def patient_visits_to_dataframe(patients_db, patient_code):
    """Extract saved visits for one Patient ID from patients_history.json."""
    patient_record = patients_db.get(patient_code, {})
    visits = patient_record.get("visits", [])
    rows = []

    for i, visit in enumerate(visits, start=1):
        glucose_value = visit.get("glucose", visit.get("Predicted_Glucose", visit.get("predicted_glucose", None)))
        if glucose_value is None:
            continue

        visit_date = visit.get("date", visit.get("Timestamp", visit.get("timestamp", f"Visit {i}")))
        rows.append({
            "Visit": i,
            "Date": visit_date,
            "Date_Display": str(visit_date),
            "Predicted_Glucose": float(glucose_value),
            "Status": visit.get("status", visit.get("Status", "Not available")),
            "State": visit.get("state", visit.get("State", "Not available")),
            "Reference_Glucose": visit.get("reference_glucose", visit.get("Reference_Glucose", "Not saved")),
            "Clarke_Zone": visit.get("clarke_zone", visit.get("Clarke_Zone", "Not validated"))
        })

    visits_df = pd.DataFrame(rows)
    if not visits_df.empty:
        visits_df["Predicted_Glucose"] = pd.to_numeric(visits_df["Predicted_Glucose"], errors="coerce")
        visits_df = visits_df.dropna(subset=["Predicted_Glucose"])
    return visits_df


def create_patient_evolution_figure(visits_df, patient_code):
    """Create a clear, colorful, medical-looking glucose evolution curve with fasting/post-meal distinction."""
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    visits_df = visits_df.copy()
    visits_df["State_Clean"] = visits_df["State"].astype(str).str.strip()

    x = visits_df["Visit"].values
    y = visits_df["Predicted_Glucose"].values
    y_max = max(260, float(max(y)) + 38)

    # Medical interpretation zones for readability
    ax.axhspan(0, 70, color="#DBEAFE", alpha=0.55, label="Low range")
    ax.axhspan(70, 140, color="#DCFCE7", alpha=0.68, label="Normal range")
    ax.axhspan(140, 200, color="#FEF3C7", alpha=0.76, label="Elevated range")
    ax.axhspan(200, y_max, color="#FEE2E2", alpha=0.66, label="High range")

    # Main evolution line connecting all visits chronologically
    ax.plot(
        x, y,
        linewidth=3.8,
        color="#7C3AED",
        alpha=0.88,
        zorder=3,
        label="Glucose evolution"
    )

    # Soft area under the curve
    ax.fill_between(x, y, [min(y) - 10] * len(y), color="#8B5CF6", alpha=0.09, zorder=2)

    # Different visual markers for fasting vs post-meal
    marker_styles = {
        "Fasting": {"color": "#2563EB", "marker": "s", "label": "Fasting measurement"},
        "Post-meal": {"color": "#EC4899", "marker": "o", "label": "Post-meal measurement"},
        "Post meal": {"color": "#EC4899", "marker": "o", "label": "Post-meal measurement"},
        "Postmeal": {"color": "#EC4899", "marker": "o", "label": "Post-meal measurement"},
    }

    plotted_labels = set()
    for _, row in visits_df.iterrows():
        state = str(row["State_Clean"])
        style = marker_styles.get(state, {"color": "#64748B", "marker": "D", "label": "Other measurement"})
        label = style["label"] if style["label"] not in plotted_labels else None
        plotted_labels.add(style["label"])

        ax.scatter(
            row["Visit"], row["Predicted_Glucose"],
            s=170,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=2.6,
            zorder=6,
            label=label
        )

    # Value + state labels above each point
    for _, row in visits_df.iterrows():
        state = str(row["State_Clean"])
        ax.annotate(
            f"{row['Predicted_Glucose']:.1f} mg/dL\n{state}",
            (row["Visit"], row["Predicted_Glucose"]),
            textcoords="offset points",
            xytext=(0, 18),
            ha="center",
            fontsize=10.2,
            fontweight="bold",
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="#CBD5E1", alpha=0.97)
        )

    # Trend line when at least 2 visits exist
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle="--", linewidth=2.5, color="#0EA5E9", alpha=0.9, label="Overall trend")

    # Date + state directly under every point so the professor understands the context
    x_labels = [f"{d}\n{s}" for d, s in zip(visits_df["Date_Display"], visits_df["State_Clean"])]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9.8)

    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12, fontweight="bold", color="#334155")
    ax.set_xlabel("Visit date and measurement state", fontsize=12, fontweight="bold", color="#334155")
    ax.set_title(f"Patient Glucose Evolution — {patient_code}", fontsize=19, fontweight="bold", color="#0F172A", pad=20)

    ax.grid(True, linestyle="--", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    return fig


def build_evolution_pdf(patient_code, patient_info, visits_df, fig):
    """Create a PDF report containing the glucose evolution curve and visit table."""

    def pdf_text(value):
        value = str(value)
        replacements = {
            "→": "->", "—": "-", "–": "-", "²": "2",
            "🩸": "", "📈": "", "📊": "", "✅": "", "⚠️": "",
            "🧠": "", "👤": "", "📄": "", "🔬": "", "💡": ""
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value.encode("latin-1", errors="replace").decode("latin-1")

    def usable_width(pdf_obj):
        """Return the printable width of the current page."""
        try:
            return pdf_obj.epw
        except Exception:
            return pdf_obj.w - pdf_obj.l_margin - pdf_obj.r_margin

    def safe_multicell(pdf_obj, height, text, border=0):
        """
        Stable replacement for multi_cell(0, ...).
        The previous version used width=0. In fpdf2, after some cell/image operations,
        the current X position can be close to the right margin, so width=0 may leave
        no horizontal space and raise:
        FPDFException: Not enough horizontal space to render a single character.
        """
        pdf_obj.set_x(pdf_obj.l_margin)
        pdf_obj.multi_cell(usable_width(pdf_obj), height, pdf_text(text), border=border)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(124, 58, 237)
    pdf.rect(0, 0, 210, 34, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 17)
    pdf.set_xy(12, 9)
    pdf.cell(0, 8, "GlucoSense AI - Patient Glucose Evolution Report", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.set_x(12)
    pdf.cell(0, 6, pdf_text(f"Patient ID: {patient_code} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True)

    pdf.ln(18)
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "Patient Summary", ln=True)
    pdf.set_font("Arial", "", 10)

    name = patient_info.get("name", "Not available")
    age = patient_info.get("age", "Not available")
    gender = patient_info.get("gender", "Not available")
    diabetes = patient_info.get("diabetes", "Not available")
    bmi = patient_info.get("bmi", "Not available")
    bmi_category = patient_info.get("bmi_category", "Not available")

    summary_text = f"""Patient: {name}
Age: {age} | Gender: {gender}
Diabetes status: {diabetes}
BMI: {bmi} ({bmi_category})
Number of visits: {len(visits_df)}"""
    safe_multicell(pdf, 6, summary_text)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=180, bbox_inches="tight")
        img_path = tmp.name

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "Glucose Evolution Curve", ln=True)
    pdf.image(img_path, x=10, w=190)

    pdf.add_page()
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Arial", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "Visit Table", ln=True)
    pdf.set_font("Arial", "", 9)

    for _, row in visits_df.iterrows():
        line = (
            f"Visit {int(row['Visit'])} | "
            f"{row['Date_Display']} | "
            f"Predicted: {row['Predicted_Glucose']:.1f} mg/dL | "
            f"Status: {row['Status']} | "
            f"State: {row['State']}"
        )
        safe_multicell(pdf, 6, line)

    pdf.ln(5)
    pdf.set_font("Arial", "I", 9)
    safe_multicell(pdf, 5, "Research disclaimer: This follow-up report is generated by a research prototype and does not replace certified glucose monitoring or professional medical advice.")

    try:
        os.remove(img_path)
    except Exception:
        pass

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)

# ── HERO ─────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div>
      <div class="hero-kicker"><span class="pulse-dot"></span> Live Biomedical AI System</div>
      <h1>🩸 GlucoSense AI</h1>
      <h2>AI-Enhanced Non-Invasive Glucose Monitoring System</h2>
      <p>Smart sensing → glucose estimation → AI health insight</p>
      <p>Intelligent glucose monitoring interface | 2026</p>
      <div class="hero-flow">
        <span class="flow-chip">1 ·  Sensor acquisition</span>
        <span class="flow-chip">2 · Glucose insight</span>
        <span class="flow-chip">3 · Optional validation</span>
        <span class="flow-chip">4 · AI report</span>
        <span class="flow-chip">5 · Longitudinal follow-up</span>
      </div>
    </div>
    <div class="hero-panel">
      <div class="hero-panel-title">System intelligence</div>
      <div class="hero-panel-value">Real-Time AI Pipeline</div>
      <p>An intelligent glucose-monitoring interface combining live sensing, glucose insight, patient context, AI interpretation, and follow-up visualization.</p>
      <div class="hero-mini-grid">
        <div class="hero-mini">📡  Sensor data stream</div>
        <div class="hero-mini">🧠  AI interpretation</div>
        <div class="hero-mini">📊  Reference validation</div>
        <div class="hero-mini">📈 Patient evolution</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────
st.sidebar.title("🧪 Measurement Context")
state_label=st.sidebar.selectbox("Measurement State",["Fasting","Post-meal"])
if state_label=="Fasting":
    State_Code=0; Time_After_Meal_min=0
else:
    State_Code=1
    Time_After_Meal_min=st.sidebar.number_input("Time After Meal (min)",min_value=0,max_value=300,value=120)
st.sidebar.markdown("---")
real_glucose=st.sidebar.number_input("Reference Glucometer (optional)",min_value=0.0,max_value=500.0,value=0.0)
validate_button=st.sidebar.button("✅ Validate with Glucometer")
refresh_button=st.sidebar.button("🔄 Refresh")
voice_enabled=st.sidebar.checkbox("🔊 Enable voice",value=True)
if refresh_button: st.rerun()

# ── LOAD DATA ─────────────────────────────────
if not os.path.exists(DATA_FILE): st.warning("No CSV found."); st.stop()
raw_df=pd.read_csv(DATA_FILE)
if raw_df.empty: st.warning("CSV empty."); st.stop()
raw_df=raw_df.tail(1).copy().replace("___GLUCOSE___",0)
for col in raw_df.columns: raw_df[col]=pd.to_numeric(raw_df[col],errors="coerce")
raw_df["State_Code"]=State_Code; raw_df["Time_After_Meal_min"]=Time_After_Meal_min

model_df=pd.DataFrame([{
    "State_Code":raw_df["State_Code"].iloc[0],"Time_After_Meal_min":raw_df["Time_After_Meal_min"].iloc[0],
    "MQ138_MinRs":raw_df["MQ138_MinRs"].iloc[0],"MQ138_AvgRs":raw_df["MQ138_AvgRs"].iloc[0],
    "MQ6_MinRs":raw_df["MQ6_MinRs"].iloc[0],"MQ6_AvgRs":raw_df["MQ6_AvgRs"].iloc[0],
    "MQ3_MinRs":raw_df["MQ3_MinRs"].iloc[0],"MQ3_AvgRs":raw_df["MQ3_AvgRs"].iloc[0],
    "IR":raw_df["IR"].iloc[0],"RED":raw_df["RED"].iloc[0],
    "SpO2":raw_df["SpO2"].iloc[0],"BPM":raw_df["BPM"].iloc[0],
    "Temperature_C":raw_df["Temperature_C"].iloc[0],"Humidity_percent":raw_df["Humidity_percent"].iloc[0],
}])
model_df=model_df[features]
prediction=float(model.predict(model_df)[0])

if state_label=="Fasting":
    if prediction<70:    status,status_class="Low Glucose","low"
    elif prediction<100: status,status_class="Normal","normal"
    elif prediction<126: status,status_class="Prediabetic Range","pre"
    else:                status,status_class="High Glucose","high"
else:
    if prediction<70:    status,status_class="Low Glucose","low"
    elif prediction<140: status,status_class="Normal","normal"
    elif prediction<200: status,status_class="Prediabetic Range","pre"
    else:                status,status_class="High Glucose","high"

last_modified=datetime.fromtimestamp(os.path.getmtime(DATA_FILE)).strftime("%Y-%m-%d %H:%M:%S")
measurement_id=f"{last_modified}_{state_label}_{Time_After_Meal_min}"

history_df=load_history()
if validate_button and real_glucose>0:
    error=prediction-real_glucose; abs_error=abs(error); zone=clarke_zone(real_glucose,prediction)
    new_row={"Measurement_ID":measurement_id,"Timestamp":last_modified,"State":state_label,
             "Time_After_Meal_min":Time_After_Meal_min,"Predicted_Glucose":round(prediction,1),
             "Status":status,"Reference_Glucose":round(real_glucose,1),
             "Error":round(error,1),"Absolute_Error":round(abs_error,1),"Clarke_Zone":zone}
    history_df=history_df[history_df["Measurement_ID"].astype(str)!=measurement_id]
    history_df=pd.concat([history_df,pd.DataFrame([new_row])],ignore_index=True)
    save_history(history_df); st.success(f"✅ Validated. Clarke Zone: {zone}")

voice_text=f"Analysis complete. Predicted glucose is {prediction:.1f} milligrams per deciliter. Status: {status}."
if voice_enabled:
    components.html(f"""<script>setTimeout(function(){{
    const m=new SpeechSynthesisUtterance("{voice_text}");
    m.lang="en-US";m.rate=0.9;window.speechSynthesis.cancel();window.speechSynthesis.speak(m);
    }},800);</script>""",height=0)
components.html(f"""<button onclick="const m=new SpeechSynthesisUtterance('{voice_text}');m.lang='en-US';m.rate=0.9;window.speechSynthesis.cancel();window.speechSynthesis.speak(m);" style="background:#ec4899;color:white;border:none;padding:12px 20px;border-radius:16px;font-size:18px;font-weight:bold;cursor:pointer;">🔊 Speak Result</button>""",height=70)

st.markdown(f'<span class="live-badge">LIVE DATA LOADED ✅ Last update: {last_modified}</span>',unsafe_allow_html=True)

c1,c2,c3=st.columns([1.4,1,1])
with c1:
    st.markdown(f"""<div class="glucose-card"><div style="font-size:18px;font-weight:700;">Predicted Glucose</div>
    <div class="glucose-value">{prediction:.1f}</div><div class="unit">mg/dL</div></div>""",unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="status-box {status_class}">{status}</div>',unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="card"><h3>Measurement Context</h3><h2>{state_label}</h2>
    <p class="value">{Time_After_Meal_min} min after meal</p>
    <p class="value">Prediction engine active ✅</p><p class="value">Live signal loaded ✅</p></div>""",unsafe_allow_html=True)

if real_glucose>0:
    error=prediction-real_glucose; abs_error=abs(error)
    accuracy=max(0,100-(abs_error/real_glucose*100)); zone=clarke_zone(real_glucose,prediction)
    st.markdown("## 📏 Glucometer Validation")
    v1,v2,v3,v4,v5=st.columns(5)
    v1.metric("Reference",f"{real_glucose:.1f} mg/dL"); v2.metric("Predicted",f"{prediction:.1f} mg/dL")
    v3.metric("Abs Error",f"{abs_error:.1f} mg/dL"); v4.metric("Approx. Agreement",f"{accuracy:.1f}%"); v5.metric("Zone",zone)

st.markdown("## 📊 Glucose Range Indicator")

# Dynamic glucose range indicator based on measurement context
# This is only a visual dashboard indicator.
# It does not change the prediction model, AI Agent, history, or any calculation.

if state_label == "Fasting":
    range_max = 180
    segments = [
        ("Low", 0, 70, "#38BDF8", "&lt;70"),
        ("Normal", 70, 100, "#22C55E", "70-99"),
        ("Prediabetic", 100, 126, "#F59E0B", "100-125"),
        ("High", 126, 180, "#EF4444", "≥126")
    ]
else:
    range_max = 260
    segments = [
        ("Low", 0, 70, "#38BDF8", "&lt;70"),
        ("Normal", 70, 140, "#22C55E", "70-139"),
        ("Prediabetic", 140, 200, "#F59E0B", "140-199"),
        ("High", 200, 260, "#EF4444", "≥200")
    ]

marker_position = min(max((prediction / range_max) * 100, 0), 100)

segments_html = ""
labels_html = ""

for label, start_value, end_value, color, limit_text in segments:
    width = ((end_value - start_value) / range_max) * 100
    segments_html += f"""
    <div style="width:{width}%; background:{color}; height:18px;"></div>
    """

    labels_html += f"""
    <div style="width:{width}%; text-align:center; font-weight:900; color:#334155; font-size:14px;">
        {label}<br>
        <span style="font-size:12px; color:#64748B;">{limit_text}</span>
    </div>
    """

range_indicator_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
</head>
<body style="margin:0; padding:0; background:transparent; font-family:Inter, Arial, sans-serif;">
<div style="
    margin-top:12px;
    margin-bottom:8px;
    padding:22px;
    border-radius:24px;
    background:rgba(255,255,255,0.78);
    border:1px solid rgba(148,163,184,0.24);
    box-shadow:0 18px 45px rgba(15,23,42,0.10);
">

    <div style="
        margin-bottom:12px;
        font-weight:900;
        color:#0F172A;
        font-size:18px;
    ">
        {state_label} thresholds
    </div>

    <div style="
        position:relative;
        width:100%;
        height:18px;
        display:flex;
        overflow:visible;
        border-radius:999px;
        box-shadow:inset 0 0 0 1px rgba(15,23,42,0.10);
    ">
        {segments_html}

        <div style="
            position:absolute;
            left:{marker_position}%;
            top:-10px;
            transform:translateX(-50%);
            width:5px;
            height:38px;
            background:#0F172A;
            border-radius:999px;
            box-shadow:0 0 14px rgba(15,23,42,0.25);
        "></div>
    </div>

    <div style="display:flex; margin-top:14px;">
        {labels_html}
    </div>

</div>
</body>
</html>
"""

# components.html renders the range bar as real HTML instead of showing raw HTML text.
components.html(range_indicator_html, height=170, scrolling=False)

MQ138_MinRs=model_df["MQ138_MinRs"].iloc[0];MQ138_AvgRs=model_df["MQ138_AvgRs"].iloc[0]
MQ6_MinRs=model_df["MQ6_MinRs"].iloc[0];MQ6_AvgRs=model_df["MQ6_AvgRs"].iloc[0]
MQ3_MinRs=model_df["MQ3_MinRs"].iloc[0];MQ3_AvgRs=model_df["MQ3_AvgRs"].iloc[0]
IR=model_df["IR"].iloc[0];RED=model_df["RED"].iloc[0];BPM=model_df["BPM"].iloc[0]
SpO2=model_df["SpO2"].iloc[0];Temperature_C=model_df["Temperature_C"].iloc[0];Humidity_percent=model_df["Humidity_percent"].iloc[0]

st.markdown("## 📡 Sensor Inputs")
s1,s2,s3=st.columns(3)
with s1:
    st.markdown(f"""<div class="card"><div class="sensor-title">MQ138</div>
    <p class="value">MinRs:{MQ138_MinRs:.1f}Ω AvgRs:{MQ138_AvgRs:.1f}Ω</p></div>""",unsafe_allow_html=True)
with s2:
    st.markdown(f"""<div class="card"><div class="sensor-title">MQ6</div>
    <p class="value">MinRs:{MQ6_MinRs:.1f}Ω AvgRs:{MQ6_AvgRs:.1f}Ω</p></div>""",unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class="card"><div class="sensor-title">MQ3</div>
    <p class="value">MinRs:{MQ3_MinRs:.1f}Ω AvgRs:{MQ3_AvgRs:.1f}Ω</p></div>""",unsafe_allow_html=True)
s4,s5=st.columns(2)
with s4:
    st.markdown(f"""<div class="card"><div class="sensor-title">MAX30102</div>
    <p class="value">IR:{IR:.0f} RED:{RED:.0f} BPM:{BPM:.1f} SpO₂:{SpO2:.1f}%</p></div>""",unsafe_allow_html=True)
with s5:
    st.markdown(f"""<div class="card"><div class="sensor-title">DHT22</div>
    <p class="value">Temp:{Temperature_C:.1f}°C Humidity:{Humidity_percent:.1f}%</p></div>""",unsafe_allow_html=True)

history_df=load_history();validated_df=history_df.dropna(subset=["Reference_Glucose"])
if not validated_df.empty:
    st.markdown("## 🧭 Clarke Error Grid")
    fig,ax=plt.subplots(figsize=(7,7))
    ax.scatter(validated_df["Reference_Glucose"],validated_df["Predicted_Glucose"],s=80)
    ax.plot([0,300],[0,300],"--",label="Ideal");ax.plot([0,300],[0,360],":",label="+20%");ax.plot([0,300],[0,240],":",label="-20%")
    for _,row in validated_df.iterrows():
        ax.text(row["Reference_Glucose"]+2,row["Predicted_Glucose"]+2,str(row["Clarke_Zone"]),fontsize=10)
    ax.set_xlim(0,300);ax.set_ylim(0,300);ax.set_xlabel("Reference (mg/dL)");ax.set_ylabel("Predicted (mg/dL)")
    ax.set_title("Clarke Error Grid");ax.grid(True);ax.legend();st.pyplot(fig)

st.markdown("## 🧾 Session History")
st.dataframe(load_history().tail(20),use_container_width=True)
st.markdown("## 📋 Processed Signal Snapshot")
st.dataframe(model_df,use_container_width=True)

# ══════════════════════════════════════════════════════
# AI AGENT — FULL SCREEN MODAL REPORT
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="font-size:30px;font-weight:900;color:#1e293b;margin:30px 0 20px;">
🧠 GlucoSense AI Agent — Clinical Research Report
</div>
""",unsafe_allow_html=True)

patients_db=load_patients()

with st.container():
    st.markdown("""
    <div style="background:rgba(255,255,255,0.96);border-radius:30px;padding:32px;
    box-shadow:0 18px 55px rgba(15,23,42,0.18);border:1px solid rgba(255,255,255,0.55);margin-bottom:24px;">
    <div style="font-size:22px;font-weight:900;color:#111827;margin-bottom:8px;">👤 Patient Registration</div>
    <div style="background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;padding:12px 16px;border-radius:12px;margin-bottom:18px;font-size:14px;">
    The AI assistant uses the patient profile for personalized report generation only. 
    </div>
    """, unsafe_allow_html=True)

    pf1,pf2,pf3,pf4=st.columns(4)
    with pf1:
        patient_code=st.text_input("Patient Code",placeholder="e.g. PT001",help="Same code = AI remembers all previous visits")
        patient_name=st.text_input("Patient Name / Label",placeholder="e.g. Patient PT001 or Imen")
        patient_age=st.number_input("Age (years)",min_value=1,max_value=120,value=25)
    with pf2:
        patient_gender=st.selectbox("Gender",["Female","Male","Prefer not to say"])
        patient_weight=st.number_input("Weight (kg)",min_value=20,max_value=300,value=65)
        patient_height=st.number_input("Height (cm)",min_value=100,max_value=230,value=165)
    with pf3:
        patient_diabetes=st.selectbox("Diabetes status",[
            "No known diabetes",
            "Prediabetes",
            "Type 1 diabetes",
            "Type 2 diabetes",
            "Unknown / Not diagnosed"
        ])
        patient_family_history=st.selectbox("Family history of diabetes",[
            "No",
            "Yes",
            "Unknown"
        ], help="Select Yes if a close family member, such as father, mother, brother, or sister, has diabetes.")
        patient_activity=st.selectbox("Physical activity level",[
            "Low",
            "Moderate",
            "High",
            "Unknown"
        ])
    with pf4:
        patient_notes=st.text_area("Additional notes (optional)",placeholder="e.g. fasting since morning, post-exercise, symptoms, etc.",height=170)

    calculated_bmi = calculate_bmi(patient_weight, patient_height)
    bmi_category = classify_bmi(calculated_bmi)
    if calculated_bmi is not None:
        st.info(f"📊 Automatically calculated BMI: {calculated_bmi:.1f} kg/m² — {bmi_category}")

    generate_btn=st.button("🧠 Generate AI-Assisted Report",type="primary",use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

if generate_btn:
    if not patient_code.strip():
        st.error("Please enter a Patient Code.")
    elif not patient_name.strip():
        st.error("Please enter the patient name.")
    else:
        calculated_bmi = calculate_bmi(patient_weight, patient_height)
        bmi_category = classify_bmi(calculated_bmi)
        patient_info={
            "code":patient_code.strip(),"name":patient_name.strip(),
            "age":patient_age,"gender":patient_gender,
            "weight":patient_weight,"height":patient_height,
            "bmi":round(calculated_bmi,1) if calculated_bmi is not None else "Not available",
            "bmi_category":bmi_category,
            "diabetes":patient_diabetes,
            "family_history":patient_family_history,
            "physical_activity":patient_activity,
            "notes":patient_notes.strip()
        }
        existing=patients_db.get(patient_code.strip(),{})
        visits=existing.get("visits",[])

        if real_glucose>0:
            ai_abs_error=abs(prediction-real_glucose)
            ai_accuracy=max(0,100-(ai_abs_error/real_glucose*100))
            ai_zone=clarke_zone(real_glucose,prediction)
        else:
            ai_abs_error=0; ai_accuracy=0; ai_zone="Not validated"

        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f172a,#312e81,#7c2d12);color:white;padding:20px 24px;border-radius:22px;margin:12px 0 20px;box-shadow:0 18px 45px rgba(15,23,42,.25);font-weight:800;">
        🧠 Initializing Clinical AI... &nbsp; 📊 Processing biomarkers... &nbsp; 🩸 Building premium report...
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("🧠 The AI assistant is preparing your comprehensive report — this may take 15-20 seconds..."):
            prompt=build_premium_prompt(patient_info,prediction,status,state_label,
                                        Time_After_Meal_min,visits,real_glucose,
                                        ai_abs_error,ai_zone,ai_accuracy)
            raw_report,err=call_gemini(prompt)

        if err:
            st.error(f"AI Agent error: {err}")
        elif not raw_report or len(raw_report.strip())<100:
            st.error("Report too short. Please try again.")
        else:
            new_visit={"date":datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "glucose":round(prediction,1),"status":status,"state":state_label,"report":raw_report}
            visits.append(new_visit)
            patients_db[patient_code.strip()]={**patient_info,"visits":visits}
            save_patients(patients_db)

            if len(visits)>=2:
                prev_glucose=float(visits[-2]["glucose"])
                diff=prediction-prev_glucose
                abs_diff=abs(diff)

                if diff>5:
                    change_text=f"Glucose increased {diff:.1f} mg/dL vs last visit"
                elif diff<-5:
                    change_text=f"Glucose decreased {abs_diff:.1f} mg/dL vs last visit"
                else:
                    change_text=f"Glucose changed only {diff:.1f} mg/dL vs last visit"

                # Priority 1: current glucose range comes before direction of change.
                # A decrease is not always improvement, and an increase is not always a problem.
                if status_class=="low":
                    tc,tb,tl,te="#60a5fa","rgba(59,130,246,0.18)","🔵 Low Glucose Alert",f"{change_text}. Current value is below the normal range; confirm with a certified glucometer."

                elif status_class=="normal":
                    if prediction < 80:
                        tc,tb,tl,te="#93c5fd","rgba(96,165,250,0.18)","🔵 Low-Normal Monitoring",f"{change_text}. Value remains normal but is close to the lower glucose range."
                    elif abs_diff<=5:
                        tc,tb,tl,te="#fde047","rgba(250,204,21,0.18)","➡️ Stable",f"Glucose changed only {diff:.1f} mg/dL vs last visit and remains within the expected range."
                    else:
                        tc,tb,tl,te="#38bdf8","rgba(56,189,248,0.18)","📊 Normal Variation",f"{change_text}, but the current value remains within the expected normal range."

                elif status_class=="pre":
                    if diff<-5:
                        tc,tb,tl,te="#4ade80","rgba(34,197,94,0.18)","📉 Improving Toward Normal",f"{change_text}, but the value is still in the borderline range and should be monitored."
                    elif diff>5:
                        tc,tb,tl,te="#f87171","rgba(239,68,68,0.18)","📈 Needs Attention",f"{change_text} and the value is in the borderline range."
                    else:
                        tc,tb,tl,te="#facc15","rgba(250,204,21,0.18)","⚡ Monitor",f"{change_text}. Value remains in the borderline range."

                elif status_class=="high":
                    if diff<-5:
                        tc,tb,tl,te="#facc15","rgba(250,204,21,0.18)","📉 Improving, Still High",f"{change_text}, but the value remains high and should be confirmed."
                    else:
                        tc,tb,tl,te="#f87171","rgba(239,68,68,0.18)","📈 Needs Attention",f"{change_text} and the value is in the high glucose range."

                else:
                    tc,tb,tl,te="#94a3b8","rgba(148,163,184,0.12)","📊 Trend Available",change_text
            else:
                tc,tb,tl,te="#94a3b8","rgba(148,163,184,0.12)","🆕 First Visit","Baseline established"

            chip_bg={"normal":"background:linear-gradient(135deg,#15803d,#22c55e)","pre":"background:linear-gradient(135deg,#b45309,#f59e0b)","high":"background:linear-gradient(135deg,#b91c1c,#ef4444)","low":"background:linear-gradient(135deg,#1d4ed8,#60a5fa)"}.get(status_class,"background:#334155")

            if status_class in("high","low"): conf_val,conf_note="⚠️ Alert","Glucose outside normal range"
            elif status_class=="pre":         conf_val,conf_note="⚡ Monitor","Borderline — monitor closely"
            else:                             conf_val,conf_note="✅ Good","Within normal range"
            if real_glucose>0 and ai_zone=="A": conf_val,conf_note="🎯 High","Zone A + low error"

            val_card_val = f"{real_glucose:.1f} mg/dL" if real_glucose>0 else "Not provided"
            val_card_note = f"Error: {ai_abs_error:.1f} mg/dL | Zone {ai_zone}" if real_glucose>0 else "No glucometer used"

            report_html=enhance_ai_report_html(simple_markdown_to_html(raw_report))
            now_str=datetime.now().strftime("%d %B %Y — %H:%M")
            session_id=f"GS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            total_visits=len(visits)

            modal_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
  * {{ margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif; }}
  body {{ background:transparent; }}
  @keyframes fadeInUp {{ from{{opacity:0;transform:translateY(40px);}} to{{opacity:1;transform:translateY(0);}} }}
  @keyframes glowPulse {{
    0%,100%{{box-shadow:0 0 40px rgba(139,92,246,0.5),0 0 80px rgba(99,102,241,0.25),0 30px 90px rgba(15,23,42,0.6);}}
    50%{{box-shadow:0 0 70px rgba(139,92,246,0.8),0 0 140px rgba(99,102,241,0.4),0 30px 90px rgba(15,23,42,0.6);}}
  }}
  @keyframes shimmer {{ 0%{{background-position:0% 50%;}} 50%{{background-position:100% 50%;}} 100%{{background-position:0% 50%;}} }}
  @keyframes slideInSection {{ from{{opacity:0;transform:translateX(-20px);}} to{{opacity:1;transform:translateX(0);}} }}
  .modal-overlay {{ position:fixed;top:0;left:0;width:100vw;height:100vh;background:radial-gradient(circle at 10% 10%,rgba(236,72,153,.25),transparent 24%),radial-gradient(circle at 90% 20%,rgba(14,165,233,.28),transparent 25%),radial-gradient(circle at 50% 95%,rgba(16,185,129,.18),transparent 26%),rgba(2,4,15,0.94);backdrop-filter:blur(18px);z-index:99999;display:flex;align-items:flex-start;justify-content:center;padding:20px;overflow-y:auto;animation:fadeInUp 0.5s ease; }}
  .modal-box {{ width:100%;max-width:1280px;background:linear-gradient(135deg,#22d3ee,#8b5cf6,#ec4899,#22c55e,#22d3ee);background-size:350% 350%;border-radius:38px;padding:4px;animation:glowPulse 4s infinite, shimmer 8s linear infinite;margin:auto; }}
  .modal-inner {{ background:radial-gradient(circle at top left,rgba(59,130,246,.18),transparent 26%),radial-gradient(circle at top right,rgba(236,72,153,.16),transparent 24%),linear-gradient(180deg,#090d22,#050816 60%,#020617);border-radius:34px;padding:56px 60px;border:1px solid rgba(255,255,255,0.10); }}
  .close-btn {{ position:absolute;top:20px;right:24px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);color:white;width:44px;height:44px;border-radius:50%;font-size:20px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s; }}
  .close-btn:hover{{background:rgba(239,68,68,0.3);border-color:rgba(239,68,68,0.5);}}
  .report-header {{ display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:28px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:32px;position:relative; }}
  .report-kicker {{ font-size:11px;font-weight:800;letter-spacing:3px;color:#60a5fa;text-transform:uppercase;margin-bottom:10px; }}
  .report-title {{ font-size:52px;font-weight:950;line-height:1.05;background:linear-gradient(90deg,#67e8f9,#a78bfa,#f0abfc,#34d399,#67e8f9);background-size:350%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 5s linear infinite;text-shadow:0 0 35px rgba(103,232,249,.18); }}
  .report-subtitle {{ font-size:14px;color:#94a3b8;margin-top:10px;max-width:600px;line-height:1.6; }}
  .gemini-badge {{ background:linear-gradient(135deg,#7c3aed,#4f46e5,#0ea5e9);color:white;padding:12px 22px;border-radius:999px;font-weight:800;font-size:13px;white-space:nowrap;box-shadow:0 8px 30px rgba(124,58,237,0.45);border:1px solid rgba(255,255,255,0.15); }}
  .patient-strip {{ display:flex;flex-wrap:wrap;gap:12px;background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.035));border:1px solid rgba(255,255,255,0.12);border-radius:26px;padding:22px 26px;margin-bottom:34px;box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 18px 45px rgba(0,0,0,.25); }}
  .pitem {{ padding:0 14px;border-right:1px solid rgba(255,255,255,0.08); }}
  .pitem:last-child{{border-right:none;}}
  .pitem-label{{font-size:10px;color:rgba(255,255,255,0.38);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;}}
  .pitem-val{{font-size:14px;font-weight:700;color:#f1f5f9;}}
  .cards-grid {{ display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:32px; }}
  .info-card {{ border-radius:26px;padding:24px;background:linear-gradient(145deg,rgba(255,255,255,0.075),rgba(255,255,255,0.025));border:1px solid rgba(255,255,255,0.13);transition:all 0.25s;box-shadow:0 18px 45px rgba(0,0,0,.22);position:relative;overflow:hidden; }}
  .info-card::before {{ content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6); }}
  .info-card:hover{{transform:translateY(-6px) scale(1.01);border-color:rgba(255,255,255,.22);}}
  .info-card .lbl{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.8px;font-weight:800;margin-bottom:10px;}}
  .info-card .val{{font-size:26px;font-weight:900;color:white;line-height:1.1;}}
  .info-card .sub{{font-size:12px;color:#94a3b8;margin-top:8px;}}
  .trend-badge {{ display:inline-block;padding:6px 16px;border-radius:999px;font-size:13px;font-weight:800;background:{tb};color:{tc};border:1px solid {tc}44; }}
  .section-divider {{ border:none;border-top:1px solid rgba(255,255,255,0.06);margin:28px 0; }}
  .ai-content-box {{ background:transparent;border:none;border-radius:0;padding:0;animation:slideInSection 0.5s ease; }}
  .ai-content-box h2 {{ margin:34px 0 0;padding:20px 24px;border-radius:24px 24px 0 0;font-size:23px;font-weight:950;color:white;letter-spacing:.4px;border:1px solid rgba(255,255,255,.12);border-bottom:none;box-shadow:0 18px 50px rgba(0,0,0,.25); }}
  .ai-content-box h2:first-child{{margin-top:0;}}
  .ai-content-box h2 span{{display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:16px;background:rgba(255,255,255,.16);margin-right:12px;box-shadow:inset 0 1px 0 rgba(255,255,255,.18);}}
  .sec-summary{{background:linear-gradient(135deg,#1d4ed8,#06b6d4);}}
  .sec-patient{{background:linear-gradient(135deg,#6d28d9,#ec4899);}}
  .sec-glucose{{background:linear-gradient(135deg,#047857,#22c55e);}}
  .sec-validation{{background:linear-gradient(135deg,#92400e,#f59e0b);}}
  .sec-trend{{background:linear-gradient(135deg,#312e81,#8b5cf6);}}
  .sec-guidance{{background:linear-gradient(135deg,#0f766e,#14b8a6);}}
  .sec-limits{{background:linear-gradient(135deg,#374151,#64748b);}}
  .sec-disclaimer{{background:linear-gradient(135deg,#7f1d1d,#ef4444);}}
  .ai-content-box h3 {{color:#c4b5fd;font-size:18px;margin:20px 0 8px;}}
  .ai-content-box p, .ai-content-box ul, .ai-content-box ol {{ background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.032));border-left:1px solid rgba(255,255,255,.10);border-right:1px solid rgba(255,255,255,.10);color:#e2e8f0;font-size:16px;line-height:1.95;margin:0;padding:18px 24px; }}
  .ai-content-box h2 + p, .ai-content-box h2 + ul, .ai-content-box h2 + ol {{ border-top:none; }}
  .ai-content-box p:last-child, .ai-content-box ul:last-child, .ai-content-box ol:last-child {{ border-radius:0 0 24px 24px;border-bottom:1px solid rgba(255,255,255,.10);box-shadow:0 18px 50px rgba(0,0,0,.20);margin-bottom:8px; }}
  .ai-content-box li{{margin:10px 0;}}
  .ai-content-box strong{{color:#ffffff;font-weight:800;}}
  .disclaimer-box {{ background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.22);color:#fde68a;padding:18px 22px;border-radius:18px;margin-top:24px;font-size:14px;line-height:1.75; }}
  .report-footer {{ display:flex;justify-content:space-between;align-items:center;border-top:1px solid rgba(255,255,255,0.06);padding-top:18px;margin-top:24px;font-size:12px;color:#475569; }}
  .close-bottom-btn {{ display:block;width:100%;background:linear-gradient(135deg,#7c3aed,#2563eb);color:white;border:none;padding:16px;border-radius:16px;font-size:16px;font-weight:800;cursor:pointer;margin-top:24px;transition:opacity 0.2s; }}
  .close-bottom-btn:hover{{opacity:0.85;}}
</style>
</head>
<body>
<div class="modal-overlay" id="modal" onclick="handleOverlayClick(event)">
  <div class="modal-box">
    <div class="modal-inner" style="position:relative;">
      <button class="close-btn" onclick="closeModal()">✕</button>
      <div class="report-header">
        <div>
          <div class="report-kicker">AI-Assisted Clinical Research Report · Session #{total_visits} · {session_id}</div>
          <div class="report-title">🏥 GlucoSense AI Agent Report</div>
          <div class="report-subtitle">AI-assisted report generated from live biomarker signals, glucose estimation, optional validation, and patient profile context.</div>
        </div>
        <div class="gemini-badge">✨ AI Health Assistant</div>
      </div>
      <div class="patient-strip">
        <div class="pitem"><div class="pitem-label">Patient</div><div class="pitem-val">{escape(patient_info['name'])}</div></div>
        <div class="pitem"><div class="pitem-label">Code</div><div class="pitem-val">{escape(patient_info['code'])}</div></div>
        <div class="pitem"><div class="pitem-label">Age</div><div class="pitem-val">{patient_info['age']} yrs</div></div>
        <div class="pitem"><div class="pitem-label">Gender</div><div class="pitem-val">{escape(patient_info['gender'])}</div></div>
        <div class="pitem"><div class="pitem-label">Weight</div><div class="pitem-val">{patient_info['weight']} kg</div></div>
        <div class="pitem"><div class="pitem-label">Height</div><div class="pitem-val">{patient_info['height']} cm</div></div>
        <div class="pitem"><div class="pitem-label">BMI</div><div class="pitem-val">{patient_info['bmi']}</div></div>
        <div class="pitem"><div class="pitem-label">BMI Class</div><div class="pitem-val">{escape(str(patient_info['bmi_category']))}</div></div>
        <div class="pitem"><div class="pitem-label">Diabetes</div><div class="pitem-val">{escape(patient_info['diabetes'])}</div></div>
        <div class="pitem"><div class="pitem-label">Family History</div><div class="pitem-val">{escape(patient_info['family_history'])}</div></div>
        <div class="pitem"><div class="pitem-label">Activity</div><div class="pitem-val">{escape(patient_info['physical_activity'])}</div></div>
        <div class="pitem"><div class="pitem-label">Visit</div><div class="pitem-val" style="color:#c084fc;">#{total_visits}</div></div>
        <div class="pitem"><div class="pitem-label">Date</div><div class="pitem-val">{now_str}</div></div>
      </div>
      <div class="cards-grid">
        <div class="info-card" style="{chip_bg};">
          <div class="lbl">Predicted Glucose</div>
          <div class="val">{prediction:.1f}</div>
          <div class="sub">mg/dL · {status}</div>
        </div>
        <div class="info-card">
          <div class="lbl">Reference Glucometer</div>
          <div class="val" style="font-size:20px;">{val_card_val}</div>
          <div class="sub">{val_card_note}</div>
        </div>
        <div class="info-card" style="background:{tb};">
          <div class="lbl">Health Trend</div>
          <div class="val" style="font-size:18px;"><span class="trend-badge">{tl}</span></div>
          <div class="sub" style="color:{tc}88;">{te}</div>
        </div>
        <div class="info-card">
          <div class="lbl">AI Assessment</div>
          <div class="val" style="font-size:20px;">{conf_val}</div>
          <div class="sub">{conf_note}</div>
        </div>
      </div>
      <hr class="section-divider">
      <div class="ai-content-box">{report_html}</div>
      <div class="disclaimer-box">
        ⚠️ <strong>Research Prototype Disclaimer:</strong> This report is generated by GlucoSense AI Agent for academic research and educational demonstration purposes only. It does not constitute medical diagnosis, does not prescribe any treatment, and must not replace certified glucose monitoring devices or professional healthcare consultation. All findings require confirmation with a calibrated glucometer and a qualified medical professional before any clinical decision is made.
      </div>
      <div class="report-footer">
        <span>Generated by GlucoSense AI Agent | Non-Invasive Glucose Monitoring Intelligent glucose monitoring interface | 2026</span>
        <span>{now_str}</span>
      </div>
      <button class="close-bottom-btn" onclick="closeModal()">✕ Close Report</button>
    </div>
  </div>
</div>
<script>
function closeModal() {{ document.getElementById('modal').style.display='none'; }}
function handleOverlayClick(e) {{ if(e.target.id==='modal') closeModal(); }}
document.addEventListener('keydown',function(e){{ if(e.key==='Escape') closeModal(); }});
</script>
</body>
</html>
"""
            components.html(modal_html, height=900, scrolling=True)

            # ─────────────────────────────────────────────
            # PDF DOWNLOAD REPORT (replaces the old TXT download)
            # ─────────────────────────────────────────────
            def pdf_safe(value):
                """Remove emojis/special characters that classic PDF fonts cannot render."""
                value = str(value)
                replacements = {
                    "🧠": "[AI]", "🩸": "[Glucose]", "📊": "[Validation]", "📈": "[Trend]",
                    "💡": "[Guidance]", "⚙️": "[Technical]", "⚠️": "[Warning]", "✅": "[OK]",
                    "🎯": "[Assessment]", "🏥": "[Clinical]", "→": "->", "—": "-", "–": "-",
                    "•": "-", "²": "2"
                }
                for a, b in replacements.items():
                    value = value.replace(a, b)
                return value.encode("latin-1", errors="replace").decode("latin-1")

            def clean_markdown_for_pdf(text):
                """Convert the AI markdown report into readable PDF text."""
                text = str(text)
                text = text.replace("## ", "\n")
                text = text.replace("### ", "\n")
                text = text.replace("**", "")
                return pdf_safe(text)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Header banner
            pdf.set_fill_color(124, 58, 237)
            pdf.rect(0, 0, 210, 36, "F")
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", "B", 18)
            pdf.set_xy(12, 9)
            pdf.cell(0, 8, "GlucoSense AI Clinical Report", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.set_x(12)
            pdf.cell(0, 6, "AI-Assisted Non-Invasive Glucose Monitoring Research Prototype", ln=True)
            pdf.set_x(12)
            pdf.cell(0, 6, pdf_safe(f"Session: {session_id} | Generated: {now_str}"), ln=True)

            pdf.ln(18)

            # Patient profile box
            pdf.set_text_color(30, 41, 59)
            pdf.set_fill_color(245, 243, 255)
            pdf.set_draw_color(124, 58, 237)
            pdf.set_line_width(0.4)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Patient Profile", border=1, ln=True, fill=True)
            pdf.set_font("Arial", "", 10)

            patient_summary = f"""
Patient: {patient_info['name']} | Code: {patient_info['code']}
Age: {patient_info['age']} years | Gender: {patient_info['gender']}
Weight: {patient_info['weight']} kg | Height: {patient_info['height']} cm | BMI: {patient_info['bmi']} ({patient_info['bmi_category']})
Diabetes status: {patient_info['diabetes']} | Family history: {patient_info['family_history']} | Physical activity: {patient_info['physical_activity']}
"""
            pdf.multi_cell(0, 6, pdf_safe(patient_summary), border=1)

            pdf.ln(4)

            # Measurement summary box
            pdf.set_fill_color(224, 242, 254)
            pdf.set_draw_color(14, 165, 233)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Measurement Summary", border=1, ln=True, fill=True)
            pdf.set_font("Arial", "", 10)

            measurement_summary = f"""
Measurement state: {state_label}
Time after meal: {Time_After_Meal_min} min
Predicted glucose: {prediction:.1f} mg/dL
Status: {status}
Reference glucometer: {val_card_val}
Validation: {val_card_note}
AI assessment: {conf_val} - {conf_note}
Health trend: {tl} - {te}
"""
            pdf.multi_cell(0, 6, pdf_safe(measurement_summary), border=1)

            pdf.ln(6)

            # AI report content
            pdf.set_fill_color(236, 253, 245)
            pdf.set_draw_color(16, 185, 129)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "AI Clinical Research Interpretation", border=1, ln=True, fill=True)
            pdf.ln(2)
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(31, 41, 55)

            clean_report = clean_markdown_for_pdf(raw_report)

            # Print paragraph by paragraph for better spacing
            for paragraph in clean_report.split("\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    pdf.ln(2)
                    continue

                # Section titles from markdown become standalone lines; make them bold
                possible_titles = [
                    "Executive Summary", "Patient-Specific Clinical Context", "Glucose Prediction Analysis",
                    "Validation and Agreement Assessment", "Longitudinal Trend Analysis",
                    "Personalized Monitoring Guidance", "System Technical Limitations",
                    "Research and Clinical Disclaimer"
                ]
                if paragraph in possible_titles:
                    pdf.ln(3)
                    pdf.set_fill_color(243, 244, 246)
                    pdf.set_text_color(88, 28, 135)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 8, paragraph, ln=True, fill=True)
                    pdf.set_text_color(31, 41, 55)
                    pdf.set_font("Arial", "", 10)
                else:
                    pdf.multi_cell(0, 5.5, paragraph)
                    pdf.ln(1)

            # Final disclaimer
            pdf.ln(5)
            pdf.set_fill_color(254, 243, 199)
            pdf.set_draw_color(245, 158, 11)
            pdf.set_text_color(120, 53, 15)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, "Research Prototype Disclaimer", border=1, ln=True, fill=True)
            pdf.set_font("Arial", "", 9)
            disclaimer_text = "This report is generated for academic research and educational demonstration purposes only. It does not constitute medical diagnosis and must not replace certified glucose monitoring devices or professional healthcare consultation."
            pdf.multi_cell(0, 5.5, pdf_safe(disclaimer_text), border=1)

            # Footer page numbers
            total_pages = pdf.page_no()
            for page_num in range(1, total_pages + 1):
                pdf.page = page_num
                pdf.set_y(-12)
                pdf.set_font("Arial", "I", 8)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(0, 8, pdf_safe(f"GlucoSense AI Report | {patient_info['code']} | Page {page_num}/{total_pages}"), align="C")

            pdf_output = pdf.output(dest="S")
            if isinstance(pdf_output, str):
                pdf_output = pdf_output.encode("latin-1", errors="replace")
            else:
                pdf_output = bytes(pdf_output)

            st.download_button(
                label="📄 Download AI Clinical Report as PDF",
                data=pdf_output,
                file_name=f"GlucoSense_AI_Report_{patient_code}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )



# ══════════════════════════════════════════════════════
# PATIENT GLUCOSE EVOLUTION CURVE — AFTER AI AGENT SAVE
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="font-size:30px;font-weight:900;color:#1e293b;margin:30px 0 20px;">
📈 Patient Glucose Evolution — Follow-up Curve
</div>
""", unsafe_allow_html=True)

patients_for_evolution = load_patients()

st.markdown("""
<div style="background:rgba(255,255,255,0.96);border-radius:30px;padding:28px;
box-shadow:0 18px 55px rgba(15,23,42,0.16);border:1px solid rgba(255,255,255,0.55);margin-bottom:24px;">
<div style="font-size:22px;font-weight:900;color:#111827;margin-bottom:8px;">📊 Longitudinal Glucose Monitoring</div>
<div style="background:#ECFDF5;border:1px solid #A7F3D0;color:#065F46;padding:12px 16px;border-radius:12px;margin-bottom:18px;font-size:14px;">
This section reads the saved Patient ID history and displays the predicted glucose evolution over multiple visits. 
</div>
""", unsafe_allow_html=True)

if not patients_for_evolution:
    st.info("No patient history found yet. Generate an AI report first to save the first patient visit.")
else:
    available_patient_ids = sorted(list(patients_for_evolution.keys()))

    default_index = 0
    try:
        if patient_code.strip() in available_patient_ids:
            default_index = available_patient_ids.index(patient_code.strip())
    except Exception:
        default_index = 0

    selected_evolution_id = st.selectbox(
        "Select Patient ID to show glucose evolution",
        available_patient_ids,
        index=default_index,
        key="selected_patient_evolution_id_after_ai"
    )

    show_curve_btn = st.button("📈 Show Evolution Curve", use_container_width=True)

    if show_curve_btn:
        visits_df = patient_visits_to_dataframe(patients_for_evolution, selected_evolution_id)
        patient_info_for_curve = patients_for_evolution.get(selected_evolution_id, {})

        if visits_df.empty:
            st.warning("No valid glucose visits found for this Patient ID.")
        else:
            latest_value = visits_df["Predicted_Glucose"].iloc[-1]
            first_value = visits_df["Predicted_Glucose"].iloc[0]
            total_change = latest_value - first_value

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Patient ID", selected_evolution_id)
            k2.metric("Total Visits", len(visits_df))
            k3.metric("Latest Glucose", f"{latest_value:.1f} mg/dL")
            k4.metric("Change vs Baseline", f"{total_change:+.1f} mg/dL")

            if len(visits_df) == 1:
                st.info("Only one visit is available. This point is considered the baseline. More measurements are needed to analyze glucose evolution.")
            elif len(visits_df) == 2:
                st.success("Two visits are available. The dashboard can show an initial comparison, but more measurements will make the trend more reliable.")
            else:
                st.success("Multiple visits are available. The curve can now show the patient glucose evolution trend.")

            fig_evolution = create_patient_evolution_figure(visits_df, selected_evolution_id)
            st.pyplot(fig_evolution, use_container_width=True)

            st.markdown("### 🧾 Visit History Used for the Curve")
            st.dataframe(
                visits_df[["Visit", "Date_Display", "Predicted_Glucose", "Status", "State", "Reference_Glucose", "Clarke_Zone"]],
                use_container_width=True
            )

            pdf_bytes = build_evolution_pdf(selected_evolution_id, patient_info_for_curve, visits_df, fig_evolution)
            st.download_button(
                label="📄 Download Patient Evolution Report as PDF",
                data=pdf_bytes,
                file_name=f"GlucoSense_Evolution_Report_{selected_evolution_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <h3>🧠 System Notice</h3>
  <p>GlucoSense AI combines live biomarker signals, glucose estimation, personalized AI interpretation, and patient follow-up visualization.</p>
</div>
""",unsafe_allow_html=True)