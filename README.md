# NetPredict

### Network Throughput Prediction & Performance Analytics

NetPredict is an end-to-end machine learning application for predicting network data throughput and analyzing network performance.

The project combines supervised and unsupervised machine learning techniques to estimate **Data Throughput (Mbps)** from network conditions and geographical information, compare multiple regression models, evaluate their predictions, and segment network observations into different performance groups using K-Means clustering.

The final system is deployed as an interactive **Streamlit dashboard**.

---

## Project Overview

Network performance depends on multiple factors such as signal strength, latency, network technology, and geographic location.

The goal of NetPredict is to build a practical machine learning system that can:

* Predict expected network throughput.
* Compare different regression algorithms.
* Evaluate model predictions on unseen test data.
* Segment network observations into performance groups.
* Provide an interactive interface for exploring predictions and model behavior.

---

## Key Features

### Throughput Prediction

Users can provide network conditions and select a regression model to estimate expected throughput in Mbps.

Supported regression models:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)
* Random Forest
* XGBoost

---

### Model Comparison

NetPredict evaluates all regression models on the same held-out test dataset using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The dashboard provides:

* Model ranking
* Metric comparison
* Performance insights
* Best-model analysis
* Model interpretation

---

### Network Performance Segmentation

K-Means clustering is used to group network observations based on:

* Signal Strength
* Data Throughput
* Latency

The resulting clusters are interpreted as:

* Low Performance
* Moderate Performance
* High Performance

These labels describe observed cluster profiles and are not intended as causal explanations.

---

### Model Evaluation

The application provides model-level evaluation including:

* Actual vs. Predicted Throughput
* Prediction Residuals
* Residual Statistics
* Residual Distribution
* MAE
* RMSE
* R²

---

## Dataset

The dataset contains **16,829 network observations** and includes network, geographic, and measurement-related features.

### Original Features

| Feature                      | Description                 |
| ---------------------------- | --------------------------- |
| Timestamp                    | Measurement timestamp       |
| Locality                     | Geographic locality         |
| Latitude                     | Geographic latitude         |
| Longitude                    | Geographic longitude        |
| Signal Strength (dBm)        | Network signal strength     |
| Signal Quality (%)           | Signal quality measurement  |
| Data Throughput (Mbps)       | Target variable             |
| Latency (ms)                 | Network latency             |
| Network Type                 | Network technology          |
| BB60C Measurement (dBm)      | Measurement from BB60C      |
| srsRAN Measurement (dBm)     | Measurement from srsRAN     |
| BladeRFxA9 Measurement (dBm) | Measurement from BladeRFxA9 |

---

## Data Preprocessing

The preprocessing workflow included:

1. Data quality inspection.
2. Removal of the constant `Signal Quality (%)` feature.
3. Removal of highly redundant measurement features.
4. Removal of `Timestamp` from the modeling features.
5. Standardization of network technology labels.
6. Separation of numerical and categorical features.
7. Train/test split using an 80/20 ratio.
8. Feature preprocessing using Scikit-learn pipelines.

### Numerical Features

* Latitude
* Longitude
* Signal Strength (dBm)
* Latency (ms)

### Categorical Features

* Locality
* Network Type

Categorical features were encoded using One-Hot Encoding, while numerical features were standardized where required.

The preprocessing pipeline ensures that transformations are learned from the training data and consistently applied during testing and deployment.

---

## Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Exploration
     │
     ▼
Data Cleaning
     │
     ▼
Feature Selection
     │
     ▼
Train / Test Split
     │
     ├──────────────────────────────┐
     ▼                              ▼
Regression Models              K-Means Clustering
     │                              │
     ▼                              ▼
Model Evaluation              Cluster Profiles
     │                              │
     └──────────────┬───────────────┘
                    ▼
             Streamlit Dashboard
```

---

## Regression Models

### Linear Regression

Used as a baseline model to establish a simple linear relationship between the input features and network throughput.

### KNN Regression

Uses neighboring observations to estimate the throughput of a new observation.

### Support Vector Regression

Uses an RBF kernel to model non-linear relationships between the input features and throughput.

### Random Forest Regression

An ensemble of decision trees that combines multiple tree predictions to produce the final regression output.

### XGBoost Regression

A gradient boosting algorithm that builds an ensemble of decision trees sequentially to improve predictive performance.

---

## Model Performance

All models were evaluated using the same held-out test dataset.

| Model             |    MAE |    RMSE |     R² |
| ----------------- | -----: | ------: | -----: |
| XGBoost           | 6.4581 | 13.3298 | 0.7359 |
| Random Forest     | 6.4640 | 13.3387 | 0.7355 |
| SVR               | 6.5950 | 13.3597 | 0.7347 |
| KNN               | 6.6088 | 13.4226 | 0.7322 |
| Linear Regression | 6.8287 | 13.3191 | 0.7363 |

### Model Selection

XGBoost was selected as the primary regression model because it achieved the **lowest MAE (6.4581 Mbps)** among the evaluated models.

However, XGBoost does not outperform every model on every metric:

* Linear Regression achieved the lowest RMSE: **13.3191**
* Linear Regression achieved the highest R²: **0.7363**
* XGBoost achieved the lowest MAE: **6.4581**

The relatively small differences between the models indicate that their overall performance is fairly close on this dataset.

---

## K-Means Performance Profiles

The clustering analysis produced three interpretable performance profiles.

| Performance Level    | Signal Strength (dBm) | Throughput (Mbps) | Latency (ms) |
| -------------------- | --------------------: | ----------------: | -----------: |
| Low Performance      |                -86.47 |              2.12 |       153.21 |
| Moderate Performance |                -91.27 |              7.56 |        77.90 |
| High Performance     |                -95.82 |             62.98 |        29.61 |

These values represent the observed cluster profiles.

The cluster labels are interpretations based on the combination of throughput, latency, and signal-related characteristics.

---

## Streamlit Dashboard

The final application provides six main sections.

### 1. Overview

Provides a high-level summary of:

* Dataset size
* Number of regression models
* Number of performance clusters
* Selected model
* Model performance
* Project objectives

### 2. Prediction

Allows users to enter:

* Locality
* Latitude
* Longitude
* Network Type
* Signal Strength
* Latency

The application then predicts expected throughput and identifies the corresponding network performance level.

### 3. Model Comparison

Provides:

* Model ranking
* MAE comparison
* RMSE comparison
* R² comparison
* Performance insights
* Best-model analysis
* Model interpretation

### 4. Network Segmentation

Provides:

* Cluster profiles
* Throughput comparison
* Latency comparison
* Interactive cluster identification

### 5. Model Evaluation

Provides:

* Model metrics
* Actual vs. predicted throughput
* Residual analysis
* Residual statistics
* Residual distribution

### 6. About

Provides information about:

* Project objectives
* Machine learning models
* Project workflow
* Technologies used

---

## Project Structure

```text
NTI-Final-Project/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── model_comparison.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Linear_Regression.ipynb
│   ├── 04_KNN.ipynb
│   ├── 05_SVR.ipynb
│   ├── 06_Random_Forest.ipynb
│   ├── 07_XGBoost.ipynb
│   ├── 08_KMeans.ipynb
│   └── 09_Model_Comparison.ipynb
│
├── models/
│   ├── linear_regression.joblib
│   ├── knn.joblib
│   ├── svr.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── kmeans.joblib
│
├── app/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

## Technologies

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Deployment / Dashboard

* Streamlit

### Development Environment

* Jupyter Notebook
* PyCharm
* Git
* GitHub

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NTI-Final-Project
```

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in the browser.

---

## Model Files

The trained models are stored in the `models/` directory.

The application loads the saved pipelines using Joblib rather than retraining the models every time the dashboard starts.

This allows the Streamlit application to focus on inference and analysis.

---

## Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted throughput.

Lower values indicate smaller average prediction errors.

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

Lower values indicate better performance.

### R² Score

Measures how much of the variance in the target variable is explained by the model.

Higher values indicate better explanatory performance.

---

## Limitations

The current project has several limitations:

* Model performance is dependent on the available dataset.
* The dataset represents specific network and geographic conditions.
* Predictions should not be interpreted as guaranteed throughput values.
* MAE represents average test-set error and is not an individual prediction confidence interval.
* K-Means cluster labels are interpretations of observed profiles rather than supervised classifications.
* The current system does not provide real-time network measurements.

---

## Future Improvements

Potential future improvements include:

* Real-time network data integration.
* Automated model retraining.
* Hyperparameter optimization and experiment tracking.
* Additional geographic visualizations.
* Monitoring model performance over time.
* Deployment to a cloud platform.
* REST API for model inference.
* Automated data pipelines.
* More advanced explainability techniques such as SHAP.

---

## Project Outcome

NetPredict demonstrates a complete machine learning workflow from:

**Data Exploration → Preprocessing → Model Development → Evaluation → Clustering → Deployment**

The project combines multiple machine learning approaches with an interactive dashboard to provide both predictive and analytical insights into network performance.

---

## Author

**Mohamed Abdullah**

Computer Science Student
AI & Machine Learning Enthusiast

---

## License

This project was developed as an educational and portfolio project.
