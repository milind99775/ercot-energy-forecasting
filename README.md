# ERCOT Grid Locational Marginal Price (LMP) & Load Forecasting POC

Developed by **Milind Verma**  
*A complete end-to-end Machine Learning Proof of Concept (POC) for electrical grid load and price forecasting.*

---

## 📌 Project Overview
Predicting power grid load and Locational Marginal Pricing (LMP)—specifically Day-Ahead Prices (DALMP) and Real-Time Prices (RTLMP)—is essential for power generators, retail providers, and market traders to mitigate risk and optimize distribution.

This project implements a comprehensive modeling pipeline that processes historic grid variables (from the **ERCOT Texas energy market**) and generates multi-step forecasts. It comparatively evaluates traditional statistical models, tree-based machine learning ensembles, and deep learning architectures.

---

## 🛠️ Key Technical Implementations

### 1. Robust Time-Series Metrics Engine
Standard regression metrics (like MSE/MAE) do not tell the whole story in time-series environments. This project features a custom evaluation framework containing **30+ statistical metrics** categorized as:
*   **Scale-Dependent Metrics**: MSE, RMSE, MAE, MedAE
*   **Percentage-Error Metrics**: MAPE, SMAPE, MAAPE
*   **Relative & Bounded Metrics**: MRAE, GMRAE, MBRAE, UMBRAE
*   **Scaled Metrics**: MASE, RMSSE
*   **Directional Accuracy**: Mean Directional Accuracy (MDA) and Forecast Bias

### 2. Feature Engineering & Preprocessing
*   **Imputation & Cleaning**: Automated handling of missing and corrupted entries via structured time-slice mean imputations.
*   **Cyclical Time Feature Encoding**: Transforming temporal markers (day of year) into sine and cosine vectors to accurately convey periodic calendar fluctuations to algorithms.
*   **Dimensionality Reduction**: Principal Component Analysis (PCA) configured to preserve 95% of training feature variance.
*   **Standardization**: Careful application of `StandardScaler` fitted *only* on training data to strictly prevent data leakage.

### 3. Comprehensive Model Selection & Benchmarking
The codebase constructs and contrasts a wide variety of predictive models:
*   **Baselines**: Naive Mean, Naive Seasonal Drift
*   **Statistical Time Series**: Simple Exponential Smoothing (SES), Holt-Winter's (HWES), Autoregression (AR), ARIMA, and SARIMAX (tuned using `pmdarima` grid search)
*   **Linear & Regularized Regressors**: Bayesian Ridge, Lasso, Ridge, ElasticNet
*   **Machine Learning Trees**: Random Forest Regressor, XGBoost, and LightGBM
*   **Deep Learning (TensorFlow/Keras)**: 
    *   Stacked Artificial Neural Networks (ANN)
    *   Long Short-Term Memory (LSTM) recurrent networks built using sliding window data generators (`WINDOW_LENGTH = 24`)
*   **Ensemble Learners**: Blended architectures (e.g., XGBoost + LightGBM + LSTM) to significantly minimize residual correlations.

---

## 📊 Summary of Key Findings
*   **Time Series Univariate Modeling**: The **SARIMAX** statistical model proved to be highly robust for capturing cyclical fluctuations in Day-Ahead pricing.
*   **Multivariate Machine Learning**: Modern gradient boosting structures (**LightGBM & XGBoost**) outperformed standard linear regression baselines on complex multivariate inputs.
*   **Deep Learning Performance**: The **TensorFlow LSTM** network provided the lowest error rates on high-frequency, non-linear sequences (RTLMP) by learning temporal dependencies over sliding 24-hour windows.
*   **Hybrid Ensembling**: Blending tree models with Deep Learning (`EnsembleXG+LIGHT+TF`) consistently flattened metric error rates and reduced overall residual correlations.

---


