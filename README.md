<div align="center">

[English](README.md) | [简体中文](README_CN.md)

# Obesity Level Prediction with Explainable AI

### From prediction to explanation and actionable what-if analysis

An interactive explainable machine learning application that combines  
**Random Forest**, **SHAP**, and **counterfactual analysis** to predict obesity levels and explain individual model decisions.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://obesity-xai-prediction.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20App-F97316?style=for-the-badge)](https://www.gradio.app/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-7C3AED?style=for-the-badge)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/hee289427-wq/obesity-xai-prediction?style=social)](https://github.com/hee289427-wq/obesity-xai-prediction/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hee289427-wq/obesity-xai-prediction?style=social)](https://github.com/hee289427-wq/obesity-xai-prediction/forks)

</div>

---

## Application Demo

<p align="center">
  <img
    src="assets/demo.gif"
    alt="Obesity XAI application demonstration"
    width="950"
  >
</p>

<p align="center">
  <strong>
    Enter a profile → generate a prediction → inspect SHAP contributions →
    adjust lifestyle variables → compare the what-if outcome
  </strong>
</p>

<div align="center">

### [Launch the Live Application](https://obesity-xai-prediction.onrender.com)

The free Render instance may take a short time to wake up after a period of inactivity.

</div>

---

## Overview

Many machine learning systems provide a prediction without explaining how the decision was made.

This project goes beyond classification by answering three practical questions:

1. **What obesity level does the model predict?**
2. **Which features influenced this individual prediction?**
3. **How could the prediction change under an alternative scenario?**

The application predicts one of seven obesity-level categories using demographic, physical, dietary, and lifestyle-related features.

It then provides:

- a prediction with model confidence;
- a SHAP waterfall explanation;
- a configurable what-if scenario;
- an original-versus-counterfactual comparison.

> This application is an educational machine learning demonstration and does not constitute medical diagnosis or health advice.

---

## Why This Project Is Different

Most introductory machine learning projects stop after reporting model accuracy.

This project connects the complete workflow:

```text
Structured data
      ↓
Data preprocessing
      ↓
Random Forest classification
      ↓
Model evaluation
      ↓
SHAP local explanation
      ↓
What-if counterfactual analysis
      ↓
Interactive Gradio interface
      ↓
Cloud deployment on Render
```

The result is not only a trained model, but a publicly accessible and explainable AI application.

---

## Key Features

### Multiclass Prediction

The Random Forest classifier predicts seven obesity-level categories:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

### SHAP Local Explanation

A SHAP waterfall plot explains how each feature contributes to the prediction for one individual.

It shows:

- features that push the prediction toward the selected class;
- features that push the prediction away from the selected class;
- the relative contribution of each feature.

### What-if Counterfactual Analysis

Users can create an alternative profile by changing:

- weight;
- physical activity frequency;
- vegetable consumption;
- water intake.

The application compares the original and what-if profiles using:

- predicted obesity level;
- model confidence;
- modified feature values;
- whether the predicted class changed.

### Interactive Portfolio Interface

The Gradio interface includes:

- grouped input sections;
- responsive layout;
- model performance cards;
- confidence indicators;
- SHAP visualization;
- counterfactual comparison table;
- reset functionality;
- desktop and mobile support.

---

## Model Performance

The Random Forest model uses **200 decision trees** and was evaluated using a stratified 80/20 train-test split.

| Metric | Result |
|---|---:|
| Test Accuracy | **95.7%** |
| Weighted F1-score | **95.8%** |
| Weighted AUC — One-vs-Rest | **99.7%** |

These metrics describe performance on the current dataset and test split. They should not be interpreted as clinical validation.

---

## Project Screenshots

<details>
<summary><strong>Prediction summary</strong></summary>

<br>

<img
  src="assets/prediction_summary.png"
  alt="Prediction summary"
  width="850"
>

</details>

<details>
<summary><strong>SHAP waterfall explanation</strong></summary>

<br>

<img
  src="assets/shap_waterfall.png"
  alt="SHAP waterfall explanation"
  width="850"
>

</details>

<details>
<summary><strong>What-if counterfactual comparison</strong></summary>

<br>

<img
  src="assets/counterfactual_comparison.png"
  alt="What-if counterfactual comparison"
  width="850"
>

</details>

<details>
<summary><strong>Complete web interface</strong></summary>

<br>

<img
  src="assets/web_interface.png"
  alt="Complete Obesity XAI web interface"
  width="950"
>

</details>

---

## Dataset

This project uses the **Obesity Levels Based on Eating Habits and Physical Condition** dataset.

The model uses 16 input features covering:

| Category | Example Features |
|---|---|
| Demographic | Gender, age |
| Physical | Height, weight |
| Dietary | High-calorie food, vegetable intake, meal frequency |
| Lifestyle | Smoking, alcohol consumption, calorie monitoring |
| Activity | Physical activity, technology usage |
| Background | Family history of overweight |
| Mobility | Transportation mode |

The target variable is:

```text
NObeyesdad
```

It contains seven obesity-level classes.

For complete feature descriptions, see:

**[Dataset Documentation](data/README.md)**

---

## Explainable AI Methods

### SHAP

SHAP is used to provide a local explanation for an individual Random Forest prediction.

For every selected profile, the waterfall plot decomposes the model output into feature-level contributions.

This helps users inspect whether the model relies on meaningful factors such as:

- weight;
- family history;
- age;
- height;
- physical activity;
- dietary habits;
- water consumption.

### Feature-based What-if Analysis

The counterfactual module modifies selected features while keeping the remaining profile unchanged.

This makes it possible to examine questions such as:

> How would the prediction change if this profile had a different weight or activity level?

The module is designed to demonstrate model sensitivity and decision boundaries. It does not prescribe medical or behavioural interventions.

---

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest |
| Explainable AI | SHAP, TreeExplainer |
| Visualization | Matplotlib |
| Web Application | Gradio |
| Version Control | Git, GitHub |
| Deployment | Render |
| Development | Google Colab, Jupyter Notebook |

---

## Project Structure

```text
obesity-xai-prediction/
│
├── app.py
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
├── README_CN.md
├── LICENSE
├── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── notebooks/
│   └── obesity_xai_analysis.ipynb
│
├── assets/
│   ├── demo.gif
│   ├── web_interface.png
│   ├── prediction_summary.png
│   ├── shap_waterfall.png
│   └── counterfactual_comparison.png
│
├── docs/
│   └── Explainable Obesity Level Prediction System _ Portfolio.pdf
│
└── data/
    └── README.md
```

---

## Analysis Notebook

The complete machine learning and explainability workflow is available in:

### [Open the Analysis Notebook](notebooks/obesity_xai_analysis.ipynb)

The notebook includes:

1. dataset loading;
2. data inspection;
3. categorical feature encoding;
4. stratified train-test splitting;
5. Random Forest training;
6. Accuracy, F1-score, and AUC evaluation;
7. confusion matrix;
8. ROC curve analysis;
9. Precision-Recall curve analysis;
10. manual profile prediction;
11. SHAP local explanation;
12. what-if counterfactual analysis.

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/hee289427-wq/obesity-xai-prediction.git
cd obesity-xai-prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Confirm the dataset location

The following file must be present in the repository root:

```text
ObesityDataSet_raw_and_data_sinthetic.csv
```

### 5. Start the application

```bash
python app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://127.0.0.1:10000
```

---

## Portfolio

A project portfolio covering the system design, model evaluation, explainability methods, web interface, deployment, and personal contributions is available here:

### [View the Project Portfolio](<docs/Explainable Obesity Level Prediction System _ Portfolio.pdf>)

---

## Project Background and Personal Contribution

The initial academic study was completed as a collaborative coursework project.

This repository presents my individually developed portfolio implementation and extension of the project, including:

- model experimentation and evaluation;
- Random Forest configuration;
- reproducible analysis notebook;
- SHAP explanation integration;
- what-if counterfactual analysis;
- Gradio application development;
- user-interface redesign;
- input validation;
- GitHub repository organization;
- Render cloud deployment;
- portfolio documentation.

---

## Limitations

- The model was trained on one structured dataset and may not generalize to other populations.
- Label encoding is used for categorical variables.
- The current application retrains the model when the service starts.
- Counterfactual features are adjusted independently.
- A changed prediction does not imply a medically valid recommendation.
- The application has not undergone clinical validation.
- Render's free service may enter sleep mode after inactivity.

---

## Future Improvements

Planned improvements include:

- saving and loading a pre-trained model;
- adding automated tests;
- introducing global SHAP analysis;
- comparing multiple classification models;
- evaluating fairness across demographic groups;
- validating counterfactual feasibility;
- improving accessibility and multilingual support;
- adding continuous integration and deployment checks.

---

## Author

**何宸臻（Eric He）**

Master of Artificial Intelligence  
Universiti Malaya

- GitHub: [hee289427-wq](https://github.com/hee289427-wq)
- Live application: [Obesity XAI Web App](https://obesity-xai-prediction.onrender.com)
- Portfolio: [Project Portfolio](<docs/Explainable Obesity Level Prediction System _ Portfolio.pdf>)

---

## Support This Project

If you find this project useful, educational, or interesting:

- give the repository a ⭐;
- try the live application;
- report issues or suggestions;
- fork the project and experiment with it.

<div align="center">

### If this project helped you understand explainable AI, please consider giving it a star ⭐

[![Star this repository](https://img.shields.io/github/stars/hee289427-wq/obesity-xai-prediction?style=for-the-badge&logo=github&label=Star%20this%20project)](https://github.com/hee289427-wq/obesity-xai-prediction)

</div>

---

## License

This project is licensed under the [MIT License](LICENSE).
