import html
import os
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =========================================================
# 1) Load data and train the model
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}. "
        "Place ObesityDataSet_raw_and_data_sinthetic.csv in the same directory as app.py."
    )

df = pd.read_csv(CSV_PATH)

label_encoders = {}

for column in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr",
    average="weighted",
)

# Build the SHAP explainer only once to reduce repeated work.
shap_explainer = shap.TreeExplainer(rf)

print("Model trained successfully.")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print(f"Weighted AUC: {auc:.4f}")


# =========================================================
# 2) Display helpers
# =========================================================

CLASS_DISPLAY_NAMES = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight Level I",
    "Overweight_Level_II": "Overweight Level II",
    "Obesity_Type_I": "Obesity Type I",
    "Obesity_Type_II": "Obesity Type II",
    "Obesity_Type_III": "Obesity Type III",
}

CLASS_BADGE_CLASSES = {
    "Insufficient_Weight": "status-blue",
    "Normal_Weight": "status-green",
    "Overweight_Level_I": "status-amber",
    "Overweight_Level_II": "status-orange",
    "Obesity_Type_I": "status-red",
    "Obesity_Type_II": "status-red",
    "Obesity_Type_III": "status-red",
}


def display_class_name(class_label):
    return CLASS_DISPLAY_NAMES.get(
        class_label,
        class_label.replace("_", " "),
    )


def clamp(value, minimum, maximum):
    return max(minimum, min(float(value), maximum))


def empty_counterfactual_table():
    return pd.DataFrame(
        columns=[
            "Scenario",
            "Weight (kg)",
            "Physical Activity",
            "Vegetable Intake",
            "Water Intake",
            "Predicted Level",
            "Confidence",
        ]
    )


def empty_result_html():
    return """
    <div class="empty-state">
        <div class="empty-icon">AI</div>
        <h3>Your prediction will appear here</h3>
        <p>
            Complete the profile, adjust the optional what-if values,
            and select <strong>Predict &amp; Explain</strong>.
        </p>
    </div>
    """


def build_result_html(
    original_label,
    original_confidence,
    counterfactual_label,
    counterfactual_confidence,
    delta_weight,
    delta_faf,
    delta_fcvc,
    delta_ch2o,
):
    original_name = html.escape(display_class_name(original_label))
    counterfactual_name = html.escape(display_class_name(counterfactual_label))

    original_badge = CLASS_BADGE_CLASSES.get(original_label, "status-blue")
    counterfactual_badge = CLASS_BADGE_CLASSES.get(
        counterfactual_label,
        "status-blue",
    )

    changed = original_label != counterfactual_label

    if changed:
        change_message = (
            '<span class="change-chip changed">Prediction changed</span>'
        )
    else:
        change_message = (
            '<span class="change-chip unchanged">Prediction unchanged</span>'
        )

    return f"""
    <div class="result-summary">
        <div class="result-heading">
            <div>
                <div class="eyebrow">MODEL OUTPUT</div>
                <h2>Prediction summary</h2>
            </div>
            {change_message}
        </div>

        <div class="prediction-grid">
            <div class="prediction-card">
                <span class="card-kicker">Original profile</span>
                <span class="status-pill {original_badge}">
                    {original_name}
                </span>
                <div class="confidence-label">
                    Model confidence
                    <strong>{original_confidence:.1%}</strong>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill"
                         style="width:{original_confidence * 100:.1f}%"></div>
                </div>
            </div>

            <div class="prediction-card">
                <span class="card-kicker">What-if profile</span>
                <span class="status-pill {counterfactual_badge}">
                    {counterfactual_name}
                </span>
                <div class="confidence-label">
                    Model confidence
                    <strong>{counterfactual_confidence:.1%}</strong>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill"
                         style="width:{counterfactual_confidence * 100:.1f}%"></div>
                </div>
            </div>
        </div>

        <div class="scenario-strip">
            <div>
                <span>Weight</span>
                <strong>{float(delta_weight):+.0f} kg</strong>
            </div>
            <div>
                <span>Physical activity</span>
                <strong>{float(delta_faf):+.1f}</strong>
            </div>
            <div>
                <span>Vegetable intake</span>
                <strong>{float(delta_fcvc):+.1f}</strong>
            </div>
            <div>
                <span>Water intake</span>
                <strong>{float(delta_ch2o):+.1f}</strong>
            </div>
        </div>

        <p class="result-note">
            This application is an educational machine-learning demonstration
            and is not a medical diagnosis.
        </p>
    </div>
    """


# =========================================================
# 3) Input encoding and SHAP explanation
# =========================================================

def encode_user_input(user_dict):
    df_input = pd.DataFrame([user_dict])

    for column in df_input.columns:
        if column in label_encoders:
            try:
                df_input[column] = label_encoders[column].transform(
                    df_input[column]
                )
            except ValueError as exc:
                raise gr.Error(
                    f"Unsupported value for {column}: {df_input[column].iloc[0]}"
                ) from exc

    return df_input[X.columns]


def build_shap_waterfall_figure(
    model,
    X_one_row_df,
    predicted_class_code,
):
    shap_values = shap_explainer.shap_values(X_one_row_df)
    X_row = X_one_row_df.iloc[0]

    if isinstance(shap_values, list):
        feature_values = shap_values[predicted_class_code][0]
        base_value = shap_explainer.expected_value[predicted_class_code]
    else:
        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 3 and shap_array.shape[0] == 1:
            feature_values = shap_array[0, :, predicted_class_code]
            base_value = shap_explainer.expected_value[predicted_class_code]
        elif shap_array.ndim == 3 and shap_array.shape[1] == 1:
            feature_values = shap_array[predicted_class_code, 0, :]
            base_value = shap_explainer.expected_value[predicted_class_code]
        elif shap_array.ndim == 2:
            feature_values = shap_array[0, :]
            base_value = shap_explainer.expected_value
        else:
            raise ValueError(
                f"Unsupported SHAP output shape: {shap_array.shape}"
            )

    explanation = shap.Explanation(
        values=feature_values,
        base_values=base_value,
        data=X_row.values,
        feature_names=list(X_one_row_df.columns),
    )

    plt.close("all")

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False,
    )

    figure = plt.gcf()
    figure.set_size_inches(10.5, 5.8)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Feature Contributions for the Original Prediction",
        fontsize=14,
        fontweight="bold",
        x=0.5,
        y=1.02,
    )
    figure.tight_layout()

    return figure


# =========================================================
# 4) Prediction function
# =========================================================

def predict_and_explain(
    Gender,
    Age,
    Height,
    Weight,
    family_history_with_overweight,
    FAVC,
    FCVC,
    NCP,
    CAEC,
    SMOKE,
    CH2O,
    SCC,
    FAF,
    TUE,
    CALC,
    MTRANS,
    delta_weight,
    delta_faf,
    delta_fcvc,
    delta_ch2o,
):
    required_values = {
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "Vegetable consumption": FCVC,
        "Number of main meals": NCP,
        "Water intake": CH2O,
        "Physical activity": FAF,
        "Technology usage": TUE,
    }

    for field_name, value in required_values.items():
        if value is None:
            raise gr.Error(f"Please enter a value for {field_name}.")

    if float(Age) <= 0:
        raise gr.Error("Age must be greater than 0.")

    if float(Height) <= 0:
        raise gr.Error("Height must be greater than 0.")

    if float(Weight) <= 0:
        raise gr.Error("Weight must be greater than 0.")

    person_original = {
        "Gender": Gender,
        "Age": float(Age),
        "Height": float(Height),
        "Weight": float(Weight),
        "family_history_with_overweight":
            family_history_with_overweight,
        "FAVC": FAVC,
        "FCVC": float(FCVC),
        "NCP": float(NCP),
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": float(CH2O),
        "SCC": SCC,
        "FAF": float(FAF),
        "TUE": float(TUE),
        "CALC": CALC,
        "MTRANS": MTRANS,
    }

    X_original = encode_user_input(person_original)

    original_code = int(rf.predict(X_original)[0])
    original_label = label_encoders[
        "NObeyesdad"
    ].inverse_transform([original_code])[0]

    original_probabilities = rf.predict_proba(X_original)[0]
    original_class_index = int(
        np.where(rf.classes_ == original_code)[0][0]
    )
    original_confidence = float(
        original_probabilities[original_class_index]
    )

    shap_figure = build_shap_waterfall_figure(
        rf,
        X_original,
        original_code,
    )

    original_weight = float(Weight)
    original_faf = float(FAF)
    original_fcvc = float(FCVC)
    original_ch2o = float(CH2O)

    counterfactual_weight = max(
        1.0,
        original_weight + float(delta_weight),
    )
    counterfactual_faf = clamp(
        original_faf + float(delta_faf),
        0,
        3,
    )
    counterfactual_fcvc = clamp(
        original_fcvc + float(delta_fcvc),
        1,
        3,
    )
    counterfactual_ch2o = clamp(
        original_ch2o + float(delta_ch2o),
        1,
        3,
    )

    person_counterfactual = person_original.copy()
    person_counterfactual["Weight"] = counterfactual_weight
    person_counterfactual["FAF"] = counterfactual_faf
    person_counterfactual["FCVC"] = counterfactual_fcvc
    person_counterfactual["CH2O"] = counterfactual_ch2o

    X_counterfactual = encode_user_input(person_counterfactual)

    counterfactual_code = int(
        rf.predict(X_counterfactual)[0]
    )
    counterfactual_label = label_encoders[
        "NObeyesdad"
    ].inverse_transform([counterfactual_code])[0]

    counterfactual_probabilities = rf.predict_proba(
        X_counterfactual
    )[0]
    counterfactual_class_index = int(
        np.where(rf.classes_ == counterfactual_code)[0][0]
    )
    counterfactual_confidence = float(
        counterfactual_probabilities[counterfactual_class_index]
    )

    comparison_table = pd.DataFrame(
        {
            "Scenario": [
                "Original",
                "What-if",
            ],
            "Weight (kg)": [
                round(original_weight, 2),
                round(counterfactual_weight, 2),
            ],
            "Physical Activity": [
                round(original_faf, 2),
                round(counterfactual_faf, 2),
            ],
            "Vegetable Intake": [
                round(original_fcvc, 2),
                round(counterfactual_fcvc, 2),
            ],
            "Water Intake": [
                round(original_ch2o, 2),
                round(counterfactual_ch2o, 2),
            ],
            "Predicted Level": [
                display_class_name(original_label),
                display_class_name(counterfactual_label),
            ],
            "Confidence": [
                f"{original_confidence:.1%}",
                f"{counterfactual_confidence:.1%}",
            ],
        }
    )

    result_html = build_result_html(
        original_label=original_label,
        original_confidence=original_confidence,
        counterfactual_label=counterfactual_label,
        counterfactual_confidence=counterfactual_confidence,
        delta_weight=delta_weight,
        delta_faf=delta_faf,
        delta_fcvc=delta_fcvc,
        delta_ch2o=delta_ch2o,
    )

    return result_html, shap_figure, comparison_table


def reset_interface():
    return (
        "Female",
        25,
        1.70,
        70,
        "no",
        "no",
        2,
        3,
        "Sometimes",
        "no",
        2,
        "no",
        1,
        1,
        "Sometimes",
        "Walking",
        0,
        0,
        0,
        0,
        empty_result_html(),
        None,
        empty_counterfactual_table(),
    )


# =========================================================
# 5) Portfolio UI
# =========================================================

CUSTOM_CSS = """
:root {
    --brand-1: #4f46e5;
    --brand-2: #7c3aed;
    --brand-3: #2563eb;
    --ink: #0f172a;
    --muted: #64748b;
    --surface: #ffffff;
    --border: #e2e8f0;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 22px 22px 40px !important;
    background:
        radial-gradient(circle at top left,
            rgba(79, 70, 229, 0.11), transparent 32%),
        radial-gradient(circle at top right,
            rgba(37, 99, 235, 0.10), transparent 26%),
        #f8fafc;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 38px;
    margin-bottom: 18px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 26px;
    color: #ffffff;
    background:
        linear-gradient(135deg,
            rgba(15, 23, 42, 0.98),
            rgba(49, 46, 129, 0.96) 54%,
            rgba(37, 99, 235, 0.94));
    box-shadow: 0 22px 60px rgba(30, 41, 59, 0.20);
}

.hero-shell::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -75px;
    top: -95px;
    border-radius: 999px;
    border: 48px solid rgba(255, 255, 255, 0.07);
}

.hero-eyebrow {
    margin-bottom: 10px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c7d2fe;
}

.hero-shell h1 {
    max-width: 920px;
    margin: 0;
    font-size: clamp(32px, 4vw, 54px);
    line-height: 1.05;
    letter-spacing: -0.035em;
    color: #ffffff;
    text-shadow: 0 3px 18px rgba(0, 0, 0, 0.22);
}

.hero-shell p {
    max-width: 780px;
    margin: 16px 0 0;
    font-size: 16px;
    line-height: 1.7;
    color: rgba(255, 255, 255, 0.83);
}

.tech-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
}

.tech-pill {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, 0.17);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(6px);
    font-size: 12px;
    font-weight: 700;
    color: #ffffff;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 0 0 18px;
}

.metric-card {
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.90);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}

.metric-card span {
    display: block;
    margin-bottom: 7px;
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-card strong {
    font-size: 24px;
    color: var(--ink);
}

.panel-card {
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    background: rgba(255, 255, 255, 0.94) !important;
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.07) !important;
}

.panel-title h2 {
    margin: 0 0 2px !important;
    color: var(--ink);
    font-size: 22px !important;
}

.panel-title p {
    margin: 0 0 14px !important;
    color: var(--muted);
    font-size: 14px;
}

.primary-action button {
    min-height: 50px !important;
    border: 0 !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    background:
        linear-gradient(135deg,
            var(--brand-1),
            var(--brand-2)) !important;
    box-shadow: 0 12px 24px rgba(79, 70, 229, 0.24) !important;
}

.secondary-action button {
    min-height: 50px !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
}

.result-summary {
    padding: 21px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background:
        linear-gradient(180deg, #ffffff, #fbfdff);
}

.result-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
}

.result-heading h2 {
    margin: 2px 0 0;
    color: var(--ink);
    font-size: 23px;
}

.eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.14em;
    color: var(--brand-1);
}

.change-chip {
    display: inline-flex;
    padding: 7px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
}

.change-chip.changed {
    color: #7c2d12;
    background: #ffedd5;
}

.change-chip.unchanged {
    color: #166534;
    background: #dcfce7;
}

.prediction-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.prediction-card {
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: #ffffff;
}

.card-kicker {
    display: block;
    margin-bottom: 10px;
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
}

.status-pill {
    display: inline-flex;
    padding: 8px 11px;
    margin-bottom: 13px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 800;
}

.status-green {
    color: #166534;
    background: #dcfce7;
}

.status-blue {
    color: #1d4ed8;
    background: #dbeafe;
}

.status-amber {
    color: #92400e;
    background: #fef3c7;
}

.status-orange {
    color: #9a3412;
    background: #ffedd5;
}

.status-red {
    color: #991b1b;
    background: #fee2e2;
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-size: 12px;
    color: var(--muted);
}

.confidence-label strong {
    color: var(--ink);
}

.confidence-track {
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: #eef2ff;
}

.confidence-fill {
    height: 100%;
    min-width: 3px;
    border-radius: 999px;
    background:
        linear-gradient(90deg,
            var(--brand-1),
            var(--brand-3));
}

.scenario-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    margin-top: 13px;
}

.scenario-strip div {
    padding: 11px;
    border-radius: 12px;
    background: #f8fafc;
}

.scenario-strip span {
    display: block;
    margin-bottom: 4px;
    font-size: 11px;
    color: var(--muted);
}

.scenario-strip strong {
    color: var(--ink);
    font-size: 13px;
}

.result-note {
    margin: 14px 0 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.55;
}

.empty-state {
    min-height: 252px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 28px;
    text-align: center;
    border: 1px dashed #cbd5e1;
    border-radius: 18px;
    background:
        linear-gradient(180deg, #ffffff, #f8fafc);
}

.empty-icon {
    width: 50px;
    height: 50px;
    display: grid;
    place-items: center;
    margin-bottom: 12px;
    border-radius: 15px;
    color: #ffffff;
    font-weight: 900;
    background:
        linear-gradient(135deg,
            var(--brand-1),
            var(--brand-3));
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.20);
}

.empty-state h3 {
    margin: 0 0 8px;
    color: var(--ink);
}

.empty-state p {
    max-width: 460px;
    margin: 0;
    color: var(--muted);
    line-height: 1.6;
}

.info-banner {
    padding: 15px 16px;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    background: #eff6ff;
    color: #1e3a8a;
    font-size: 13px;
    line-height: 1.6;
}

.footer-shell {
    margin-top: 18px;
    padding: 24px;
    border: 1px solid var(--border);
    border-radius: 20px;
    background: #ffffff;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
}

.footer-shell h3 {
    margin: 0 0 8px;
    color: var(--ink);
}

.footer-shell p {
    margin: 0;
    color: var(--muted);
    line-height: 1.7;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 12px 12px 28px !important;
    }

    .hero-shell {
        padding: 26px 22px;
        border-radius: 20px;
    }

    .metric-grid,
    .prediction-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .scenario-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 560px) {
    .metric-grid,
    .prediction-grid,
    .scenario-strip {
        grid-template-columns: 1fr;
    }

    .hero-shell h1 {
        font-size: 31px;
    }
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(
    title="Obesity XAI Prediction",
) as demo:
    gr.HTML(
        """
        <section class="hero-shell">
            <div class="hero-eyebrow">
                Explainable Machine Learning Portfolio Project
            </div>
            <h1>Obesity Level Prediction with Explainable AI</h1>
            <p>
                An interactive application that combines a Random Forest
                classifier, SHAP feature explanations, and counterfactual
                what-if analysis to make model decisions easier to inspect.
            </p>
            <div class="tech-row">
                <span class="tech-pill">Random Forest</span>
                <span class="tech-pill">SHAP</span>
                <span class="tech-pill">Counterfactual Analysis</span>
                <span class="tech-pill">Scikit-learn</span>
                <span class="tech-pill">Gradio</span>
                <span class="tech-pill">Render</span>
            </div>
        </section>
        """
    )

    gr.HTML(
        f"""
        <section class="metric-grid">
            <article class="metric-card">
                <span>Test accuracy</span>
                <strong>{accuracy:.1%}</strong>
            </article>
            <article class="metric-card">
                <span>Weighted F1</span>
                <strong>{f1:.1%}</strong>
            </article>
            <article class="metric-card">
                <span>Weighted AUC</span>
                <strong>{auc:.1%}</strong>
            </article>
            <article class="metric-card">
                <span>Model</span>
                <strong>RF · 200 trees</strong>
            </article>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["panel-card"]):
            gr.Markdown(
                """
                ## Build a profile
                Enter demographic and lifestyle information, then use the
                what-if controls to create an alternative scenario.
                """,
                elem_classes=["panel-title"],
            )

            with gr.Accordion(
                "1. Personal information",
                open=True,
            ):
                with gr.Row():
                    Gender = gr.Dropdown(
                        choices=["Female", "Male"],
                        value="Female",
                        label="Gender",
                    )
                    Age = gr.Number(
                        value=25,
                        label="Age",
                    )

                with gr.Row():
                    Height = gr.Number(
                        value=1.70,
                        label="Height (m)",
                    )
                    Weight = gr.Number(
                        value=70,
                        label="Weight (kg)",
                    )

                family_history_with_overweight = gr.Dropdown(
                    choices=["yes", "no"],
                    value="no",
                    label="Family history of overweight",
                )

            with gr.Accordion(
                "2. Eating habits",
                open=False,
            ):
                FAVC = gr.Dropdown(
                    choices=["yes", "no"],
                    value="no",
                    label="Frequent high-calorie food consumption",
                )

                with gr.Row():
                    FCVC = gr.Number(
                        value=2,
                        label="Vegetable consumption frequency (1–3)",
                    )
                    NCP = gr.Number(
                        value=3,
                        label="Number of main meals",
                    )

                CAEC = gr.Dropdown(
                    choices=[
                        "no",
                        "Sometimes",
                        "Frequently",
                        "Always",
                    ],
                    value="Sometimes",
                    label="Eating between meals",
                )

                with gr.Row():
                    CH2O = gr.Number(
                        value=2,
                        label="Daily water intake (1–3)",
                    )
                    CALC = gr.Dropdown(
                        choices=[
                            "no",
                            "Sometimes",
                            "Frequently",
                            "Always",
                        ],
                        value="Sometimes",
                        label="Alcohol consumption",
                    )

            with gr.Accordion(
                "3. Lifestyle and activity",
                open=False,
            ):
                with gr.Row():
                    SMOKE = gr.Dropdown(
                        choices=["yes", "no"],
                        value="no",
                        label="Smoking",
                    )
                    SCC = gr.Dropdown(
                        choices=["yes", "no"],
                        value="no",
                        label="Calorie monitoring",
                    )

                with gr.Row():
                    FAF = gr.Number(
                        value=1,
                        label="Physical activity frequency (0–3)",
                    )
                    TUE = gr.Number(
                        value=1,
                        label="Technology-device usage (0–2)",
                    )

                MTRANS = gr.Dropdown(
                    choices=[
                        "Automobile",
                        "Motorbike",
                        "Bike",
                        "Public_Transportation",
                        "Walking",
                    ],
                    value="Walking",
                    label="Primary transportation mode",
                )

            with gr.Accordion(
                "4. What-if scenario",
                open=True,
            ):
                gr.Markdown(
                    """
                    Adjusting these controls does not change the original
                    SHAP explanation. It creates a second profile for
                    counterfactual comparison.
                    """
                )

                delta_weight = gr.Slider(
                    minimum=-50,
                    maximum=50,
                    value=0,
                    step=1,
                    label="Weight change (kg)",
                )

                delta_faf = gr.Slider(
                    minimum=-3,
                    maximum=3,
                    value=0,
                    step=0.1,
                    label="Physical activity change",
                )

                delta_fcvc = gr.Slider(
                    minimum=-2,
                    maximum=2,
                    value=0,
                    step=0.1,
                    label="Vegetable consumption change",
                )

                delta_ch2o = gr.Slider(
                    minimum=-2,
                    maximum=2,
                    value=0,
                    step=0.1,
                    label="Water intake change",
                )

            with gr.Row():
                predict_button = gr.Button(
                    "Predict & Explain",
                    variant="primary",
                    elem_classes=["primary-action"],
                )
                reset_button = gr.Button(
                    "Reset",
                    variant="secondary",
                    elem_classes=["secondary-action"],
                )

        with gr.Column(scale=7, elem_classes=["panel-card"]):
            gr.Markdown(
                """
                ## Results and explanation
                Review the model output, feature contributions, and
                original-versus-what-if comparison.
                """,
                elem_classes=["panel-title"],
            )

            result_output = gr.HTML(
                value=empty_result_html(),
            )

            with gr.Accordion(
                "SHAP feature contribution plot",
                open=True,
            ):
                shap_plot_output = gr.Plot(
                    label="",
                    show_label=False,
                )

                gr.HTML(
                    """
                    <div class="info-banner">
                        <strong>How to read this plot:</strong>
                        red features push the model output higher for the
                        predicted class, while blue features push it lower.
                        The plot explains the original profile only.
                    </div>
                    """
                )

            with gr.Accordion(
                "Original vs. what-if comparison",
                open=True,
            ):
                comparison_output = gr.Dataframe(
                    value=empty_counterfactual_table(),
                    interactive=False,
                    label="",
                    show_label=False,
                )

    gr.HTML(
        """
        <section class="footer-shell">
            <h3>Project overview</h3>
            <p>
                This portfolio project demonstrates an end-to-end machine
                learning workflow: structured-data preprocessing, multiclass
                classification, model evaluation, local explainability with
                SHAP, interactive counterfactual analysis, and cloud deployment.
                Predictions are provided for educational demonstration only.
            </p>
        </section>
        """
    )

    all_inputs = [
        Gender,
        Age,
        Height,
        Weight,
        family_history_with_overweight,
        FAVC,
        FCVC,
        NCP,
        CAEC,
        SMOKE,
        CH2O,
        SCC,
        FAF,
        TUE,
        CALC,
        MTRANS,
        delta_weight,
        delta_faf,
        delta_fcvc,
        delta_ch2o,
    ]

    all_outputs = [
        result_output,
        shap_plot_output,
        comparison_output,
    ]

    predict_button.click(
        fn=predict_and_explain,
        inputs=all_inputs,
        outputs=all_outputs,
    )

    reset_button.click(
        fn=reset_interface,
        inputs=None,
        outputs=all_inputs + all_outputs,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )
