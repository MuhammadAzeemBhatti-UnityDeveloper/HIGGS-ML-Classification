# 📘 HIGGS Dataset – Machine Learning Classification  
### Logistic Regression vs Neural Network (MLP) Comparative Study

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Dataset](https://img.shields.io/badge/Dataset-HIGGS-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📌 Overview

This project presents a **full end-to-end machine learning pipeline** applied to the **HIGGS Dataset (11 million samples)** — a highly complex, real scientific dataset used in particle physics experiments.

The goal is to compare:

- **Logistic Regression (LR)**
- **Neural Network (MLPClassifier)**  

across different **training sample sizes** (1% → 100%).

### 📊 We evaluate:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- PR-AUC  
- Training Time Scaling  

---

## 📂 Project Structure

HIGGS-ML-Classification/
│
├── notebooks/
│ ├── 01_data_loading.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_logistic_regression_experiments.ipynb
│ ├── 04_neural_network_experiments.ipynb
│ └── 05_full_pipeline.ipynb
│
├── src/
│ ├── logistic_regression.py
│ ├── neural_network.py
│ ├── preprocessing.py
│ └── utils.py
│
├── figures/
│ ├── lr_roc_curve.png
│ ├── nn_roc_curve.png
│ ├── lr_pr_curve.png
│ ├── nn_pr_curve.png
│ ├── lr_vs_nn_roc.png
│ ├── lr_vs_nn_pr.png
│ ├── training_time_lr_vs_nn.png
│ ├── comparison_f1_lr_nn.png
│ └── comparison_accuracy_lr_nn.png
│
├── reports/
│ ├── LogisticRegressionReport.txt
│ ├── NeuralNetworkReport.txt
│ └── higgs_summary.txt
│
├── README.md
└── requirements.txt

yaml
Copy code

---

## 🧪 Dataset Description

The **HIGGS dataset** consists of simulated particle collision events used in high-energy physics.

### **Dataset Properties**
- **Total Samples:** 11,000,000  
- **Features:** 28  
- **Target:** Binary (Signal = 1, Background = 0)

### **Class Balance**
| Class | Count | Ratio |
|-------|--------|--------|
| Signal (1) | ~5.83M | 52.9% |
| Background (0) | ~5.17M | 47.1% |

The dataset is **moderately balanced**, making it ideal for ROC/PR analysis.

---

## 🧹 Preprocessing Steps

✔ Loaded 11M rows in memory  
✔ Verified no missing values  
✔ Detected duplicate rows  
✔ Applied **StandardScaler**  
✔ Split dataset using **stratified 80/20**  
✔ Generated statistical summary (`higgs_summary.txt`)

---

## 🔬 Experiment Settings

Models trained using:

1%, 2%, 3%, 4%, 5%, 10%, 15%, 20%, 30%,
40%, 50%, 60%, 70%, 80%, 90%, 100%

yaml
Copy code

Metrics collected:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Training Time  

Additional full-data curves:

- ROC Curve  
- Precision–Recall Curve  

---

# 📈 Results Summary

## 🔵 Logistic Regression (Full dataset)
- **Accuracy:** 0.641  
- **F1-score:** 0.686  
- **ROC-AUC:** 0.684  
- **PR-AUC:** 0.683  
- **Training Time:** ~86 sec  

## 🟣 Neural Network (Full dataset)
- **Accuracy:** 0.768  
- **F1-score:** 0.785  
- **ROC-AUC:** 0.852  
- **PR-AUC:** 0.865  
- **Training Time:** ~3200 sec  

---

# 📊 Visualizations

> Upload your images into the **figures/** folder  
> They will appear automatically in the README

---

## 📉 Performance Scaling

<details>
<summary><strong>Logistic Regression Plots</strong></summary>

![LR Performance](plots/LR Performance vs Sample Size.png)  
![LR Time](plots/LR Training Time vs Sample Size.png)

</details>

<details>
<summary><strong>Neural Network Plots</strong></summary>

![NN Performance](plots/NN Performance vs Sample Size.png)  
![NN Time](plots/NN Training Time vs Sample Size.png)

</details>

---

## 🆚 LR vs NN – ROC & PR Curves

<details>
<summary><strong>ROC Curve Comparison</strong></summary>

![ROC](plots/ROC Curve Comparison.png)

</details>

<details>
<summary><strong>PR Curve Comparison</strong></summary>

![PR](plots/PR Curve Comparison.png)

</details>

---

## 🛠 How to Run

### 1️⃣ Create Environment

conda create -n higgs python=3.13 -y
conda activate higgs

shell
Copy code

### 2️⃣ Install requirements

pip install -r requirements.txt

shell
Copy code

### 3️⃣ Launch Notebooks

jupyter notebook

yaml
Copy code

---

## 🧠 Technologies Used

- Python 3.13  
- Scikit-learn  
- Pandas  
- Matplotlib  
- NumPy  
- Jupyter Notebook  

---

## 📄 License

Distributed under the **MIT License**.

---

## 👤 Author

**Muhammad Azeem Bhatti**  
Machine Learning Engineer  
GitHub: https://github.com/MuhammadAzeemBhatti-UnityDeveloper

---
