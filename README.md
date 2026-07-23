<div align="center">

<img src="img.png" alt="Heart Failure Mortality Risk Analysis Banner" width="100%">

# ❤️ Heart Failure Mortality Risk Analysis

### Exploratory Data Analysis and Machine Learning for Predicting Mortality Among Heart Failure Patients

<br>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/github/license/Rishikeshsanin/Heart-Failure-Mortality-Risk-Analysis?style=for-the-badge)](LICENSE)

<br>

[📓 View Notebook](heart_failure_mortality_risk_analysis.ipynb) •
[📊 View Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) •
[🚀 Run Locally](#-getting-started)

</div>

---

## 📌 Overview

This project analyzes clinical records of patients diagnosed with heart failure and develops machine-learning models to predict the recorded **`DEATH_EVENT`** outcome.

The notebook covers the complete data-science workflow:

- Data loading and validation
- Data-quality assessment
- Exploratory Data Analysis
- Clinical feature visualization
- Correlation analysis
- Stratified train-test splitting
- Machine-learning model training
- Model comparison using multiple metrics
- Five-fold cross-validation
- ROC curve and confusion matrix analysis
- Feature-importance analysis
- Investigation of the follow-up `time` feature

> **Important:** This project predicts mortality among patients who already have heart failure. It does not predict whether a healthy person will develop heart failure.

---

## 🎯 Project Objective

The main objectives are to:

1. Explore the demographic and clinical characteristics of heart failure patients.
2. Identify features associated with the recorded mortality outcome.
3. Build multiple classification models.
4. Compare the models using suitable classification metrics.
5. select a model based primarily on ROC-AUC performance.
6. Examine whether the follow-up `time` feature creates an unrealistic performance advantage.

---

## 🔎 Project at a Glance

| Category | Details |
|---|---|
| Dataset | Heart Failure Clinical Records |
| Source | UCI Machine Learning Repository |
| Patient records | 299 |
| Dataset columns | 13 |
| Input predictors | 12 |
| Primary-model predictors | 11, excluding `time` |
| Target variable | `DEATH_EVENT` |
| Problem type | Binary classification |
| Models compared | 4 |
| Primary selected model | Random Forest |
| Test ROC-AUC | **0.780** |
| Test recall | **0.632** |
| Test F1-score | **0.571** |
| Mean 5-fold CV ROC-AUC | **0.781 ± 0.074** |

---

## 🗂️ Dataset

The project uses the **Heart Failure Clinical Records Dataset** from the UCI Machine Learning Repository.

The notebook searches for the dataset locally using either of these paths:

```text
heart_failure_clinical_records_dataset.csv
data/heart_failure_clinical_records_dataset.csv
```

When the CSV is not available locally, the notebook automatically downloads it from UCI.

### Target Variable

| Value | Meaning |
|---:|---|
| `0` | Patient survived during the recorded follow-up period |
| `1` | Patient death was recorded during the follow-up period |

### Clinical Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `anaemia` | Whether the patient had anaemia |
| `creatinine_phosphokinase` | Level of the CPK enzyme in the blood |
| `diabetes` | Whether the patient had diabetes |
| `ejection_fraction` | Percentage of blood leaving the heart during contraction |
| `high_blood_pressure` | Whether the patient had hypertension |
| `platelets` | Platelet count in the blood |
| `serum_creatinine` | Serum creatinine level |
| `serum_sodium` | Serum sodium level |
| `sex` | Recorded biological sex |
| `smoking` | Whether the patient smoked |
| `time` | Follow-up duration |
| `DEATH_EVENT` | Target outcome |

---

## 🧭 Analysis Workflow

```text
Heart Failure Clinical Records
              │
              ▼
      Data Loading & Validation
              │
              ▼
       Data-Quality Checks
              │
              ▼
 Exploratory Data Analysis (EDA)
              │
              ▼
    Correlation & Risk Patterns
              │
              ▼
 Primary Feature Set Without Time
              │
              ▼
     Stratified Train-Test Split
              │
              ▼
      Train Four ML Models
              │
              ▼
   Test Metrics + Cross-Validation
              │
              ▼
  ROC Curves & Confusion Matrix
              │
              ▼
      Feature-Importance Analysis
              │
              ▼
 Sensitivity Check Including Time
```

---

## 🔬 Exploratory Data Analysis

The notebook investigates:

- Dataset dimensions and column types
- Missing values
- Duplicate records
- Summary statistics
- Target-class distribution
- Skewness and extreme clinical measurements
- Mortality rates across binary clinical features
- Numeric-feature distributions by outcome
- Correlation between features and `DEATH_EVENT`

### Outlier Decision

Extreme clinical values were **not automatically deleted**.

In medical data, unusually high or low measurements may represent genuine high-risk patients rather than data-entry errors. Removing them without medical justification could distort the analysis and remove clinically important records.

<div align="center">

<img src="img2.png" alt="Heart Failure Medical Illustration" width="650">

</div>

---

## 🧠 Machine-Learning Models

Four classification models were trained and evaluated:

| Model | Important Configuration |
|---|---|
| Logistic Regression | Standard scaling and balanced class weights |
| Decision Tree | Maximum depth of 4 and balanced class weights |
| Random Forest | 400 trees, minimum leaf size of 2 and balanced class weights |
| Gradient Boosting | Gradient boosting classifier with a fixed random state |

For Logistic Regression, scaling is performed inside a Scikit-learn pipeline to prevent preprocessing leakage.

---

## ⚙️ Modeling Strategy

### Primary Feature Set

The primary models exclude the `time` column:

```python
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]
```

The dataset is divided using an 80/20 stratified split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification preserves the target-class proportions across the training and testing sets.

### Why Is `time` Excluded?

The `time` feature represents the duration for which the patient was followed.

Although it can strongly improve model performance, it may not be available when performing an initial patient-risk assessment. Therefore:

- The main models exclude `time`.
- A separate sensitivity experiment measures its effect.
- Results involving `time` are interpreted cautiously.

---

## 📊 Model Performance

### Test-Set Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **0.700** | **0.522** | **0.632** | **0.571** | **0.780** |
| Gradient Boosting | 0.717 | 0.571 | 0.421 | 0.485 | 0.750 |
| Decision Tree | 0.717 | 0.538 | **0.737** | **0.622** | 0.744 |
| Logistic Regression | 0.700 | 0.526 | 0.526 | 0.526 | 0.734 |

### Five-Fold Cross-Validation

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| **Random Forest** | **0.781** | **0.074** |
| Logistic Regression | 0.769 | 0.055 |
| Gradient Boosting | 0.724 | 0.079 |
| Decision Tree | 0.653 | 0.122 |

---

## 🏆 Selected Model

The **Random Forest classifier** was selected because it achieved:

```text
Test ROC-AUC:                 0.780
Test accuracy:                0.700
Test recall for deaths:       0.632
Test F1-score:                0.571
Mean cross-validated ROC-AUC: 0.781 ± 0.074
```

The Decision Tree produced a higher recall of `0.737`, but Random Forest provided the strongest ROC-AUC performance on both the held-out test set and cross-validation.

The selected model is therefore the best-performing model **within this experiment**, not a clinically validated model.

---

## 🔑 Feature Importance

The Random Forest model ranked the following features as the most influential:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Serum creatinine | 0.2444 |
| 2 | Ejection fraction | 0.2031 |
| 3 | Creatinine phosphokinase | 0.1187 |
| 4 | Age | 0.1182 |
| 5 | Platelets | 0.1129 |
| 6 | Serum sodium | 0.0968 |

These values describe how the trained Random Forest used the features. They do not establish that any feature medically causes mortality.

---

## ⏱️ Effect of the Follow-Up `time` Feature

A separate Logistic Regression experiment compared performance with and without `time`.

| Feature Set | Mean CV ROC-AUC | Standard Deviation |
|---|---:|---:|
| Without `time` | 0.769 | 0.055 |
| With `time` | 0.877 | 0.033 |

Including `time` substantially increased the score. However, this improvement must be treated cautiously because follow-up duration may contain information that would not be available during an initial assessment.

For this reason, the model excluding `time` remains the primary and more realistic analysis.

---

## 📏 Evaluation Metrics

The models are evaluated using:

- **Accuracy** — proportion of all predictions that were correct
- **Precision** — proportion of predicted deaths that were actual deaths
- **Recall** — proportion of recorded deaths identified by the model
- **F1-score** — harmonic mean of precision and recall
- **ROC-AUC** — ability to rank positive cases above negative cases across thresholds
- **Confusion matrix** — counts of correct and incorrect predictions by class
- **Five-fold cross-validation** — evaluation across multiple stratified dataset splits

Recall is particularly useful in this analysis because false negatives represent recorded deaths that the model failed to identify.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Interactive analysis environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine-learning models and evaluation |
| Git & GitHub | Version control and project hosting |

---

## 📁 Repository Structure

```text
Heart-Failure-Mortality-Risk-Analysis/
│
├── heart_failure_mortality_risk_analysis.ipynb
│   └── Complete EDA and machine-learning notebook
│
├── img.png
│   └── Project banner
│
├── img2.png
│   └── Medical illustration
│
├── README.md
│   └── Project documentation
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Files excluded from version control
│
└── LICENSE
    └── MIT License
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Rishikeshsanin/Heart-Failure-Mortality-Risk-Analysis.git
```

### 2. Enter the Project Directory

```bash
cd Heart-Failure-Mortality-Risk-Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
heart_failure_mortality_risk_analysis.ipynb
```

Then select:

```text
Kernel → Restart Kernel and Run All Cells
```

An internet connection is required when the dataset is not available locally.

---

## 📦 Dependencies

The project uses:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
ipython
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Limitations

- The dataset contains only 299 patient records.
- Performance estimates may vary across different train-test splits.
- The records represent patients who already had heart failure.
- The dataset is observational and cannot establish medical causation.
- Feature importance is specific to the trained model.
- The test set contains only 60 records.
- The models have not been externally validated.
- No model in this project should be used for diagnosis or treatment decisions.
- Model performance may not generalize to other hospitals, regions or patient populations.

---

## 🔮 Future Improvements

- [ ] Perform systematic hyperparameter tuning
- [ ] Add nested cross-validation
- [ ] Optimize the classification threshold based on recall and precision
- [ ] Evaluate probability calibration
- [ ] Add SHAP-based model explanations
- [ ] Validate the model using an independent dataset
- [ ] Build a Streamlit demonstration interface
- [ ] Create a FastAPI prediction service
- [ ] Add automated testing and a GitHub Actions workflow

---

## 🩺 Medical Disclaimer

This repository is intended strictly for **educational and research purposes**.

It is not a medical device, diagnostic system or clinical decision-support tool. The predictions and observations must not be used to diagnose, treat or manage any medical condition.

Always consult qualified healthcare professionals for medical guidance.

---

## 👨‍💻 Author

### Rishikesh Munnaluri

Computer Science Engineering student interested in software development, data science and machine learning.

[![GitHub](https://img.shields.io/badge/GitHub-Rishikeshsanin-181717?style=for-the-badge&logo=github)](https://github.com/Rishikeshsanin)

---

## 📄 License

This project is available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) for providing the dataset
- Scikit-learn, Pandas, NumPy, Matplotlib and Seaborn for the open-source tools used in this project

---

<div align="center">

### ⭐ Found this project useful?

Consider giving the repository a star.

**Made with Python, Jupyter Notebook and Machine Learning**

</div>
