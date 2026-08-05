import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import gradio as gr

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



# 1) Load data + Train model


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}. "
        "Upload ObesityDataSet_raw_and_data_sinthetic.csv to the Space root directory."
    )

df = pd.read_csv(CSV_PATH)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Metrics (optional)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

print("Model trained.")
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("AUC (multi-class):", auc)



# 2) Helper: encode user input

def encode_user_input(user_dict):
    """
    user_dict: human-readable dict (strings for categorical)
    return: df_encoded with same feature order as X
    """
    df_input = pd.DataFrame([user_dict])

    # Encode categorical columns using same encoders
    for col in df_input.columns:
        if col in label_encoders:
            df_input[col] = label_encoders[col].transform(df_input[col])

    # Ensure correct column order
    df_input = df_input[X.columns]
    return df_input



# 3) Helper: Build SHAP waterfall figure (robust)

def build_shap_waterfall_figure(model, X_one_row_df, pred_class_code):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_one_row_df)

    X_row = X_one_row_df.iloc[0]

    # --- Case 1: list output (multi-class old style) ---
    if isinstance(shap_values, list):
        sv = shap_values[pred_class_code][0]  # (n_features,)
        base_value = explainer.expected_value[pred_class_code]

    # --- Case 2: ndarray output ---
    else:
        arr = np.array(shap_values)

        # Your case: (1, n_features, n_classes) e.g. (1,16,7)
        if arr.ndim == 3 and arr.shape[0] == 1:
            sv = arr[0, :, pred_class_code]   # (n_features,)
            base_value = explainer.expected_value[pred_class_code]

        # Another possible shape: (n_classes, 1, n_features)
        elif arr.ndim == 3 and arr.shape[1] == 1:
            sv = arr[pred_class_code, 0, :]   # (n_features,)
            base_value = explainer.expected_value[pred_class_code]

        # Binary/regression: (1, n_features)
        elif arr.ndim == 2:
            sv = arr[0, :]
            base_value = explainer.expected_value

        else:
            raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")

    # Build Explanation (single output vector)
    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=X_row.values,
        feature_names=list(X_one_row_df.columns)
    )

    plt.close("all")
    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    return fig



# 4) Main function for Gradio

def predict_and_explain(
    Gender, Age, Height, Weight, family_history_with_overweight,
    FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS,
    delta_weight, delta_faf, delta_fcvc, delta_ch2o
):
    # ---- Original input (human-readable)
    person_original = {
        "Gender": Gender,
        "Age": float(Age),
        "Height": float(Height),
        "Weight": float(Weight),
        "family_history_with_overweight": family_history_with_overweight,
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
        "MTRANS": MTRANS
    }

    # ---- Encode original
    X_orig_encoded = encode_user_input(person_original)

    # ---- Predict original
    pred_class_code = int(rf.predict(X_orig_encoded)[0])
    pred_label = label_encoders["NObeyesdad"].inverse_transform([pred_class_code])[0]

    # ---- SHAP figure (based on original input)
    fig = build_shap_waterfall_figure(rf, X_orig_encoded, pred_class_code)

    # ---- Counterfactual (What-if) changes
    original_weight = float(Weight)
    original_faf = float(FAF)
    original_fcvc = float(FCVC)
    original_ch2o = float(CH2O)

    cf_weight = original_weight + float(delta_weight)
    cf_faf = original_faf + float(delta_faf)
    cf_fcvc = original_fcvc + float(delta_fcvc)
    cf_ch2o = original_ch2o + float(delta_ch2o)

    # avoid invalid values
    if cf_weight < 1:
        cf_weight = 1

    # FAF typically 0–3
    if cf_faf < 0:
        cf_faf = 0
    if cf_faf > 3:
        cf_faf = 3

    # FCVC typically 1–3
    if cf_fcvc < 1:
        cf_fcvc = 1
    if cf_fcvc > 3:
        cf_fcvc = 3

    # CH2O typically 1–3
    if cf_ch2o < 1:
        cf_ch2o = 1
    if cf_ch2o > 3:
        cf_ch2o = 3

    person_cf = person_original.copy()
    person_cf["Weight"] = cf_weight
    person_cf["FAF"] = cf_faf
    person_cf["FCVC"] = cf_fcvc
    person_cf["CH2O"] = cf_ch2o

    X_cf_encoded = encode_user_input(person_cf)
    cf_class_code = int(rf.predict(X_cf_encoded)[0])
    cf_label = label_encoders["NObeyesdad"].inverse_transform([cf_class_code])[0]

    # ---- Counterfactual table
    cf_table = pd.DataFrame({
        "Scenario": ["Original", "What-if (Counterfactual)"],
        "Weight (kg)": [round(original_weight, 2), round(cf_weight, 2)],
        "FAF (0–3)": [round(original_faf, 2), round(cf_faf, 2)],
        "FCVC (1–3)": [round(original_fcvc, 2), round(cf_fcvc, 2)],
        "CH2O (1–3)": [round(original_ch2o, 2), round(cf_ch2o, 2)],
        "Predicted Obesity Level": [pred_label, cf_label]
    })

    # ---- Text output
    pred_text = (
        f"Original Prediction: {pred_label}\n"
        f"What-if changes:\n"
        f"  Weight: {delta_weight:+.0f} kg\n"
        f"  FAF: {delta_faf:+.1f}\n"
        f"  FCVC: {delta_fcvc:+.1f}\n"
        f"  CH2O: {delta_ch2o:+.1f}\n"
        f"Counterfactual Prediction: {cf_label}"
    )

    return pred_text, fig, cf_table


# =========================
# 5) Gradio UI
# =========================
with gr.Blocks(title="Obesity Prediction + SHAP + What-if Counterfactual") as demo:
    gr.Markdown("## Obesity Level Prediction (Random Forest) + SHAP Waterfall + What-if Counterfactual")

    with gr.Row():
        with gr.Column(scale=1):
            Gender = gr.Dropdown(["Female", "Male"], value="Female", label="Gender")
            Age = gr.Number(value=25, label="Age")
            Height = gr.Number(value=1.70, label="Height (m)")
            Weight = gr.Number(value=70, label="Weight (kg)")

            family_history_with_overweight = gr.Dropdown(["yes", "no"], value="no", label="Family history with overweight")
            FAVC = gr.Dropdown(["yes", "no"], value="no", label="FAVC (High caloric food frequently)")
            FCVC = gr.Number(value=2, label="FCVC (Vegetable consumption)")
            NCP = gr.Number(value=3, label="NCP (Number of main meals)")
            CAEC = gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], value="Sometimes", label="CAEC (Eating between meals)")
            SMOKE = gr.Dropdown(["yes", "no"], value="no", label="SMOKE")
            CH2O = gr.Number(value=2, label="CH2O (Water daily)")
            SCC = gr.Dropdown(["yes", "no"], value="no", label="SCC (Calories monitoring)")
            FAF = gr.Number(value=1, label="FAF (Physical activity frequency)")
            TUE = gr.Number(value=1, label="TUE (Time using technology devices)")
            CALC = gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], value="Sometimes", label="CALC (Alcohol consumption)")
            MTRANS = gr.Dropdown(
                ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
                value="Walking",
                label="MTRANS (Transportation)"
            )

            # ---- What-if sliders (Weight + 3 new features)
            delta_weight = gr.Slider(
                minimum=-80,
                maximum=80,
                value=25,
                step=1,
                label="What-if Weight Change (kg)"
            )

            delta_faf = gr.Slider(
                minimum=-3,
                maximum=3,
                value=0,
                step=0.1,
                label="What-if FAF Change"
            )

            delta_fcvc = gr.Slider(
                minimum=-2,
                maximum=2,
                value=0,
                step=0.1,
                label="What-if FCVC Change"
            )

            delta_ch2o = gr.Slider(
                minimum=-2,
                maximum=2,
                value=0,
                step=0.1,
                label="What-if CH2O Change"
            )

            btn = gr.Button("Predict + Explain")

        with gr.Column(scale=1):
            pred_output = gr.Textbox(label="Prediction Output", lines=6)
            shap_plot_output = gr.Plot(label="SHAP Waterfall Plot")
            cf_table_output = gr.Dataframe(label="What-if Counterfactual Table", interactive=False)

    btn.click(
        fn=predict_and_explain,
        inputs=[
            Gender, Age, Height, Weight, family_history_with_overweight,
            FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS,
            delta_weight, delta_faf, delta_fcvc, delta_ch2o
        ],
        outputs=[pred_output, shap_plot_output, cf_table_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
