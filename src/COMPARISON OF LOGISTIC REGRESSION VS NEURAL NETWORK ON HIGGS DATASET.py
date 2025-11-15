# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 13:30:56 2025

@author: azeem
"""

import os
#import kaggle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve


#Getting your Current Working Directory
print("Current Working Directory:",os.getcwd())

#Downloading The Dataset From Kaggle
#os.system("kaggle competitions download -c santander-customer-transaction-prediction -p ./santander")

#Building The File Path
#root="C:\\MSAI\\Machine Learning\\Assignment 3\\Dataset_Satlander";
#training_path = root+"\\train.csv"

#Load the Training Data
df_train = pd.read_csv("C:\\MSAI\\Machine Learning\\Assignment 3\\HIGGS.csv", header=None)#, nrows=5_000_000, header=None
#df_train = pd.read_csv(training_path)
#df_test = pd.read_csv(test_path)

print("Dataset loaded successfully!")
print("Shape:", df_train.shape)
#print(df_train.head())

##################
#Pre- Processing
##################

# Checking Missing Values
missing_counts_train = df_train.isnull().sum()
#missing_counts_test = df_test.isnull().sum()
#print("Missing values Train per column:\n", missing_counts_train)
print("\nTotal missing values Train:", missing_counts_train.sum())
#print("\nTotal missing values Test:", missing_counts_test.sum())

# Checking Duplicates
duplicate_count_train = df_train.duplicated().sum()
print("Number of duplicated rows Train:", duplicate_count_train)
#duplicate_count_test = df_test.duplicated().sum()
#print("Number of duplicated rows Test:", duplicate_count_test)

#Check Target Class Balance
print(df_train[0].value_counts())
print(df_train[0].value_counts(normalize=True))  # Shows that the Dataset is imbalanced

#Check Outliers and If Scaling is Required
#print(df_train.describe().T)

# Save describe().T to a text file
output_path = "C:\\MSAI\\Machine Learning\\Assignment 3\\higgs_summary.txt"

with open(output_path, "w") as f:
    f.write(df_train.describe().T.to_string())

print("File saved at:", output_path)

Y = df_train[0]          # target
X = df_train.iloc[:, 1:] # 28 features

#Splitting into Training And Validation
#cannot use stratified K Fold as large number of dataset therefore we will use stratified Hold Out Cross Validation
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

#Scaling Features

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

print("Scaling complete!")
print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    """
    Fits model on training data, predicts on test data,
    and returns metrics + training time.
    """
    import time
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "train_time": train_time
    }

percentages = [1,2,3,4,5,10,15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results_lr = []

n_train = X_train_scaled.shape[0]

for p in percentages:
    frac = p / 100.0
    n_samples = int(n_train * frac)

    # Take first n_samples (you could also shuffle before, but train_test_split already shuffled)
    X_sub = X_train_scaled[:n_samples]
    y_sub = y_train.iloc[:n_samples]

    print(f"\nTraining Logistic Regression with {p}% of training data ({n_samples} samples)")

    # Basic LR model (no class_weight yet for this experiment)
    lr_clf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)


    metrics = evaluate_classifier(lr_clf, X_sub, y_sub, X_test_scaled, y_test)
    metrics["percentage"] = p
    metrics["n_samples"] = n_samples

    results_lr.append(metrics)
    
# Convert results to DataFrame
results_lr_df = pd.DataFrame(results_lr)
print("\nLogistic Regression results:")
print(results_lr_df)
# Save describe().T to a text file
output_path = "C:\\MSAI\\Machine Learning\\Assignment 3\\LogisticRegressionReport.txt"

with open(output_path, "w") as f:
    f.write(results_lr_df.to_string())

print("File saved at:", output_path)

#PLOTS: PERFORMANCE & TRAINING TIME VS SAMPLE SIZE

plt.figure(figsize=(10, 6))
plt.plot(results_lr_df["percentage"], results_lr_df["accuracy"], marker="o", label="Accuracy")
plt.plot(results_lr_df["percentage"], results_lr_df["precision"], marker="o", label="Precision")
plt.plot(results_lr_df["percentage"], results_lr_df["recall"], marker="o", label="Recall")
plt.plot(results_lr_df["percentage"], results_lr_df["f1"], marker="o", label="F1-score")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Score")
plt.title("Logistic Regression Performance vs Sample Size (Santander)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(results_lr_df["percentage"], results_lr_df["train_time"], marker="o")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Training Time (seconds)")
plt.title("Logistic Regression Training Time vs Sample Size (Santander)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Train LR on full training data for ROC/PR curves
lr_full = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1,random_state=42)
lr_full.fit(X_train_scaled, y_train)

# Probabilities
y_score_full = lr_full.predict_proba(X_test_scaled)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_test, y_score_full)
roc_auc_full = roc_auc_score(y_test, y_score_full)

# PRC
prec, rec, _ = precision_recall_curve(y_test, y_score_full)
ap_full = average_precision_score(y_test, y_score_full)

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LR (AUC = {roc_auc_full:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression on HIGGS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot PRC
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f"LR (AP = {ap_full:.3f})")
baseline = y_test.mean()
plt.hlines(baseline, 0, 1, colors='k', linestyles='--', label=f"Baseline (pos rate = {baseline:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Logistic Regression on HIGGS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#NEURAL NETWORK (MLP) – EXPERIMENTS ACROSS PERCENTAGES

results_nn = []

for p in percentages:
    frac = p / 100.0
    n_samples = int(n_train * frac)

    X_sub = X_train_scaled[:n_samples]
    y_sub = y_train.iloc[:n_samples]

    print(f"\nTraining Neural Network with {p}% of training data ({n_samples} samples)")

    nn_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,   # dataset is huge, keep it moderate
        random_state=42
    )

    metrics = evaluate_classifier(nn_clf, X_sub, y_sub, X_test_scaled, y_test)
    metrics["percentage"] = p
    metrics["n_samples"] = n_samples

    results_nn.append(metrics)

# Convert results to DataFrame
results_nn_df = pd.DataFrame(results_nn)
print("\nNeural Network results:")
print(results_nn_df)

# Save NN results to text file
nn_output_path = "C:\\MSAI\\Machine Learning\\Assignment 3\\NeuralNetworkReport.txt"
with open(nn_output_path, "w") as f:
    f.write(results_nn_df.to_string())
print("NN results saved at:", nn_output_path)

# PLOTS – NEURAL NETWORK PERFORMANCE VS SAMPLE SIZE (HIGGS)

plt.figure(figsize=(10, 6))
plt.plot(results_nn_df["percentage"], results_nn_df["accuracy"], marker="o", label="Accuracy")
plt.plot(results_nn_df["percentage"], results_nn_df["precision"], marker="o", label="Precision")
plt.plot(results_nn_df["percentage"], results_nn_df["recall"], marker="o", label="Recall")
plt.plot(results_nn_df["percentage"], results_nn_df["f1"], marker="o", label="F1-score")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Score")
plt.title("Neural Network Performance vs Sample Size (HIGGS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(results_nn_df["percentage"], results_nn_df["train_time"], marker="o")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Training Time (seconds)")
plt.title("Neural Network Training Time vs Sample Size (HIGGS)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_nn_df["percentage"], results_nn_df["roc_auc"], marker="o", label="ROC AUC")
plt.plot(results_nn_df["percentage"], results_nn_df["average_precision"], marker="o", label="Average Precision (PR AUC)")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Score")
plt.title("Neural Network ROC AUC & PR AUC vs Sample Size (HIGGS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# NEURAL NETWORK – ROC & PR CURVES ON FULL TRAINING DATA

nn_full = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)
print("\nTraining full Neural Network model on all training data for ROC/PR curves...")
nn_full.fit(X_train_scaled, y_train)

# Probabilities
y_score_nn = nn_full.predict_proba(X_test_scaled)[:, 1]

# ROC
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_score_nn)
roc_auc_nn = roc_auc_score(y_test, y_score_nn)

# PRC
prec_nn, rec_nn, _ = precision_recall_curve(y_test, y_score_nn)
ap_nn = average_precision_score(y_test, y_score_nn)

print(f"NN ROC AUC: {roc_auc_nn:.4f}")
print(f"NN AP (PR AUC): {ap_nn:.4f}")

# Combined ROC plot: LR vs NN
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LR (AUC = {roc_auc_full:.3f})")
plt.plot(fpr_nn, tpr_nn, label=f"NN (AUC = {roc_auc_nn:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – HIGGS (Logistic Regression vs Neural Network)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Combined PRC plot: LR vs NN
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f"LR (AP = {ap_full:.3f})")
plt.plot(rec_nn, prec_nn, label=f"NN (AP = {ap_nn:.3f})")
baseline = y_test.mean()
plt.hlines(baseline, 0, 1, colors='k', linestyles='--', label=f"Baseline (pos rate = {baseline:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – HIGGS (Logistic Regression vs Neural Network)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# LR VS NN – COMPARISON VS PERCENTAGE (HIGGS)

plt.figure(figsize=(10, 6))
plt.plot(results_lr_df["percentage"], results_lr_df["f1"], marker="o", label="LR F1")
plt.plot(results_nn_df["percentage"], results_nn_df["f1"], marker="o", label="NN F1")
plt.xlabel("Training Data Used (%)")
plt.ylabel("F1-score")
plt.title("F1-score vs Training Size – LR vs NN (HIGGS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_lr_df["percentage"], results_lr_df["roc_auc"], marker="o", label="LR ROC AUC")
plt.plot(results_nn_df["percentage"], results_nn_df["roc_auc"], marker="o", label="NN ROC AUC")
plt.xlabel("Training Data Used (%)")
plt.ylabel("ROC AUC")
plt.title("ROC AUC vs Training Size – LR vs NN (HIGGS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_lr_df["percentage"], results_lr_df["train_time"], marker="o", label="LR Train Time")
plt.plot(results_nn_df["percentage"], results_nn_df["train_time"], marker="o", label="NN Train Time")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time vs Training Size – LR vs NN (HIGGS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
