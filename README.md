# SignalSense: PM 980 Dataset for Signal Classification

This repository contains an end-to-end Machine Learning and Deep Learning pipeline designed to classify health statuses using time-domain
features from the **PM 980 multi-sensor dataset**. 

The project evaluates and compares various classical machine learning models alongside deep learning architectures to predict 9 distinct 
mechanical health conditions under different environment and noise levels.

## 📌 Project Overview & Specifications
The classification task is based on the **PM 980 Dataset** with the following technical characteristics and constraints:
- **Inputs:** 8 sensor streams (1 sound/voice, 3 accelerometer, 3 gyroscope, and temperature) sampled at 90Hz.
- **Output Classes (9 balanced classes):** `healthy`, `scratch`, `notchshort`, `notchlong`, `singlecutlong`, `singlecutshort`, `twocutlong`, `twocutshort`,
   and `warped`.
- **Constraint:** Strictly limited to time and frequency domain statistical features (Time-frequency analysis methods like STFT, wavelet transform, or
  MFCC are not used).

## 🛠️ Methodology & Pipeline

1. **Feature Extraction:** 
   - Statistical time-domain features (Mean, Standard Deviation, Min, Max, Median, Kurtosis, and Skewness) are computed dynamically for each of the 8 sensor columns.
   - Target classes and metadata (`environment`, `noise`) are encoded using Scikit-Learn's `LabelEncoder`.
   
2. **Data Splitting:** 
   - A deterministic `random_seed = 13` is enforced.
   - The original raw records are strictly split into **80% Training** and **20% Test** partitions before feature extraction to prevent data leakage.

3. **Model Evaluation & Cross-Validation:**
   - **Stratified 10-Fold Cross-Validation** is utilized during training to maintain proportional class distributions across all folds.
   - Performance metrics such as **Accuracy, F1-Score, Precision, Recall, Training Time, and Test Time** are tracked and compared across multiple models.

## 📊 Model Comparison & Results

The framework benchmarks classical Scikit-Learn classifiers along with state-of-the-art gradient boosting frameworks:
- **Evaluated Models:** Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, and XGBoost.
- **Top Performer:** **XGBoost** achieved the highest cross-validation accuracy (~86.88%) and solid test generalization (~80.20%).
- **Visualizations:** The notebook includes cross-correlation analysis, feature importance plots, and detailed per-class Confusion Matrices to assess structural failures.

*(Optional Deep Learning Extension)*: The codebase also features structured implementation setups for **1D CNN** and **LSTM** architectures 
targeting high-frequency sequence classification.

