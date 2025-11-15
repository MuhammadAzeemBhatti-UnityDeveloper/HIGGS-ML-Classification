📘 HIGGS Dataset – Machine Learning Classification Project
Comparative Study: Logistic Regression vs Neural Network (MLP)








📌 Project Overview

This project presents a full machine learning pipeline applied to the HIGGS Dataset (11 million samples), one of the largest public binary classification datasets.

The goal is to compare:

Logistic Regression

Neural Network (MLPClassifier)

across a wide range of training sample sizes (1% → 100%), measuring:

Accuracy

Precision

Recall

F1-score

ROC-AUC

PR-AUC

Training time scaling

This repository demonstrates expertise in:

✔ Big data handling
✔ ML modeling & evaluation
✔ Experiment scaling
✔ ROC & PR curve analysis
✔ Visualization
✔ Research-style reporting

A perfect addition to a machine learning portfolio.

📂 Repository Structure
HIGGS-ML-Classification/
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_logistic_regression_experiments.ipynb
│   ├── 04_neural_network_experiments.ipynb
│   └── 05_full_pipeline.ipynb
│
├── src/
│   ├── logistic_regression.py
│   ├── neural_network.py
│   ├── preprocessing.py
│   └── utils.py
│
├── figures/
│   ├── lr_roc_curve.png
│   ├── nn_roc_curve.png
│   ├── lr_pr_curve.png
│   ├── nn_pr_curve.png
│   ├── lr_vs_nn_roc.png
│   ├── lr_vs_nn_pr.png
│   ├── lr_time_vs_size.png
│   ├── nn_time_vs_size.png
│   ├── comparison_f1_lr_nn.png
│   └── comparison_accuracy_lr_nn.png
│
├── reports/
│   ├── LogisticRegressionReport.txt
│   ├── NeuralNetworkReport.txt
│   └── higgs_summary.txt
│
├── README.md
└── requirements.txt

📊 Dataset Description

The HIGGS Dataset contains:

11,000,000 samples

29 columns:

Column 0 → Target (1 = signal, 0 = background)

Columns 1-28 → Real-valued physics features

Class Balance:
Class	Count	Ratio
Signal (1)	~5.83M	52.9%
Background (0)	~5.17M	47.1%

The dataset is moderately balanced, making it ideal for ROC/PR analysis.

🧹 Preprocessing Steps

✔ Loaded entire 11M-row dataset
✔ Verified no missing values
✔ Checked duplicates
✔ Standard scaling of all features
✔ 80/20 stratified split
✔ Generated full statistical summary (higgs_summary.txt)

🔬 Experiment Setup

Models were trained on:

1%, 2%, 3%, 4%, 5%, 10%, 15%, 20%, 
30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%


For each percentage, we collected:

Accuracy

Precision

Recall

F1-score

Training time

Additionally:

Full ROC and PR curves generated for both models

Combined LR vs NN performance comparisons made

🚀 Key Findings
⭐ Logistic Regression (100% data)

Accuracy ≈ 0.641

F1-score ≈ 0.686

ROC-AUC ≈ 0.684

PR-AUC ≈ 0.683

Training time ≈ 86 seconds

⭐ Neural Network (100% data)

Accuracy ≈ 0.768

F1-score ≈ 0.785

ROC-AUC ≈ 0.852

PR-AUC ≈ 0.865

Training time ≈ 3200+ seconds

🧠 Final Conclusion:

Neural Networks outperform Logistic Regression in all performance metrics—but require dramatically more computation time.
Logistic Regression is fast and stable, but cannot match NN classification power on high-dimensional nonlinear data like HIGGS.

📊 Selected Visualizations

(After uploading images to figures/, these links will display graphs automatically.)

🟠 ROC Curve – LR vs NN
![ROC Curve Comparison](plots/lr_vs_nn_roc.png)

🟣 PR Curve – LR vs NN
![PR Curve Comparison](plots/lr_vs_nn_pr.png)

🔵 Training Time Comparison
![Training Time Comparison](plots/training_time_lr_vs_nn.png)

🟢 F1-score Comparison
![F1 Comparison](plots/comparison_f1_lr_nn.png)

🧪 How to Run the Code
1️⃣ Create Environment
conda create -n higgs python=3.13 -y
conda activate higgs

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Launch Notebooks
jupyter notebook

🛠 Technologies Used

Python 3.13

Scikit-learn

Pandas

Matplotlib

NumPy

Jupyter Notebook

📜 Reports Included

Inside /reports:

LogisticRegressionReport.txt

NeuralNetworkReport.txt

higgs_summary.txt (full 29-feature statistical summary)

💡 Why This Project is Portfolio-Ready

This repository demonstrates:

✔ Handling extremely large datasets
✔ Applying ML models at scale
✔ Performance benchmarking
✔ Computation–accuracy tradeoff analysis
✔ Clean code + modular structure
✔ Professional documentation

This is the kind of project that impresses hiring managers.

📄 License

This project is licensed under the MIT License.

👤 Author

Muhammad Azeem Bhatti
Machine Learning Engineer
GitHub: username
