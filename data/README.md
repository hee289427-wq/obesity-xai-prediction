# Dataset Documentation

## Dataset Name

**Obesity Levels Based on Eating Habits and Physical Condition**

This dataset is used to develop a multiclass machine learning model for obesity-level prediction.

It contains demographic, physical, dietary, and lifestyle-related information.

---

## Dataset File

The application expects the dataset file to be named:

```text
ObesityDataSet_raw_and_data_sinthetic.csv
```

For the current project structure, the CSV file is stored in the repository root:

```text
obesity-xai-prediction/
├── app.py
├── ObesityDataSet_raw_and_data_sinthetic.csv
└── data/
    └── README.md
```

Do not move the CSV file into the `data` folder unless the dataset path in `app.py` and the analysis notebook is also updated.

---

## Input Features

The dataset contains 16 input features.

| Feature | Description |
|---|---|
| `Gender` | Gender of the individual |
| `Age` | Age in years |
| `Height` | Height in metres |
| `Weight` | Weight in kilograms |
| `family_history_with_overweight` | Family history of overweight |
| `FAVC` | Frequent consumption of high-calorie food |
| `FCVC` | Frequency of vegetable consumption |
| `NCP` | Number of main meals |
| `CAEC` | Consumption of food between meals |
| `SMOKE` | Smoking status |
| `CH2O` | Daily water consumption |
| `SCC` | Monitoring of calorie consumption |
| `FAF` | Physical activity frequency |
| `TUE` | Time spent using technology devices |
| `CALC` | Alcohol consumption |
| `MTRANS` | Primary transportation mode |

---

## Target Variable

The target variable is:

```text
NObeyesdad
```

It contains seven obesity-level categories:

1. `Insufficient_Weight`
2. `Normal_Weight`
3. `Overweight_Level_I`
4. `Overweight_Level_II`
5. `Obesity_Type_I`
6. `Obesity_Type_II`
7. `Obesity_Type_III`

The task is therefore formulated as a multiclass classification problem.

---

## Data Preprocessing

The project applies the following preprocessing steps:

1. Load the CSV dataset using Pandas.
2. Separate the input features from the target variable.
3. Encode categorical variables using `LabelEncoder`.
4. Split the dataset into training and testing sets.
5. Apply an 80/20 stratified train-test split.
6. Train a Random Forest classifier.
7. Evaluate the model using Accuracy, Weighted F1-score, and Weighted AUC.

---

## Dataset Usage in the Application

The dataset is loaded in `app.py` using a path relative to the application file:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"
```

This allows the application to run locally and on Render without relying on a personal Google Drive path.

---

## Reproducibility

The complete data-processing and model-development workflow is available in:

```text
notebooks/obesity_xai_analysis.ipynb
```

The notebook includes:

- Data loading
- Label encoding
- Random Forest training
- Model performance evaluation
- Confusion matrix
- ROC curves
- Precision-Recall curves
- SHAP explanation
- What-if counterfactual analysis

---

## Responsible Use

This dataset and the resulting application are used for educational machine learning and explainable AI demonstration.

The predictions do not constitute medical diagnosis or health advice.
