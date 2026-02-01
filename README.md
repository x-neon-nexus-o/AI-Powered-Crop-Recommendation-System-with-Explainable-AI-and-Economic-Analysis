# 🌾 AI-Powered Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An intelligent crop recommendation system using ensemble machine learning models with explainable AI capabilities for sustainable agriculture decision-making.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Status](#-project-status)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notebooks Guide](#-notebooks-guide)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This semester project develops an advanced crop recommendation system that goes beyond traditional ML predictions by incorporating:

1. **Multiple ML Models** - Logistic Regression, Decision Tree, Random Forest, SVM
2. **Comprehensive EDA** - 43+ visualizations for data understanding
3. **Feature Engineering** - 39 engineered features from 7 original features
4. **Model Comparison** - Systematic comparison of all trained models
5. **Production-Ready Models** - Saved as pickle files for deployment

### Problem Statement

Farmers often struggle to select the optimal crop for their land based on soil conditions and climate. This system provides data-driven recommendations to maximize yield and profitability.

### Research Alignment

This project aligns with academic curriculum covering:
- **Unit I**: Data Collection, Cleaning, Preprocessing
- **Unit II**: Exploratory Data Analysis, Statistical Analysis
- **Unit III**: Supervised Classification Algorithms
- **Unit IV**: Ensemble Methods, Model Evaluation

---

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🔬 **Data Pipeline** | Complete ETL from raw data to ML-ready features | ✅ Complete |
| 📊 **EDA & Visualization** | 43 comprehensive plots and charts | ✅ Complete |
| 🤖 **Multiple ML Models** | 4 different classification algorithms | ✅ Complete |
| 📈 **Model Comparison** | Performance metrics across all models | ✅ Complete |
| 🎯 **High Accuracy** | Up to 99.32% test accuracy (Random Forest) | ✅ Achieved |
| 🔧 **Feature Engineering** | 39 features from original 7 | ✅ Complete |
| 💾 **Model Persistence** | All models saved as .pkl files | ✅ Complete |
| 🌐 **Web Application** | Flask-based prediction interface | ⏳ Planned |
| 📖 **Explainable AI** | SHAP-based explanations | ⏳ Planned |
| 💰 **Economic Analysis** | ROI calculations for crops | ⏳ Planned |

---

## 📊 Project Status

### ✅ Completed Components

| Component | Files | Description |
|-----------|-------|-------------|
| **Notebooks** | 9 notebooks | Data pipeline + 4 ML models |
| **Models** | 4 trained models | LR, DT, RF, SVM |
| **Visualizations** | 43 plots | EDA + Model performance |
| **Processed Data** | 22+ files | Cleaned, engineered, ML-ready |
| **Results** | 17 files | Reports, predictions, comparisons |

### ⏳ Planned Components

| Component | Status | Priority |
|-----------|--------|----------|
| XGBoost Model | Not Started | High |
| LightGBM Model | Not Started | High |
| Stacking Ensemble | Not Started | High |
| SHAP Explainability | Not Started | Medium |
| Economic Analysis | Not Started | Medium |
| Flask Web App | Not Started | Medium |
| Crop Rotation Planning | Not Started | Low |

---

## 🛠️ Technology Stack

### Core Technologies
| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Notebooks** | Jupyter Notebook 7.0.6 |
| **Data Science** | Pandas 2.1.0, NumPy 1.24.3 |
| **Machine Learning** | scikit-learn 1.3.0 |
| **Visualization** | Matplotlib 3.7.2, Seaborn 0.12.2, Plotly 5.16.1 |
| **Statistics** | SciPy 1.11.2 |
| **Web Framework** | Flask 3.0.0 (planned) |

### Planned Technologies
| Category | Technologies |
|----------|-------------|
| **Boosting** | XGBoost 1.7.6, LightGBM 4.0.0 |
| **Explainability** | SHAP 0.42.1 |

---

## 📁 Project Structure

```
crop-recommendation-system/
│
├── 📓 notebooks/                          # Jupyter Notebooks (9 implemented)
│   ├── 01_Data_Collection_and_Loading.ipynb
│   ├── 2.Data Cleaning and Preprocessing.ipynb
│   ├── 3.Exploratory Data Analysis.ipynb
│   ├── 4.Feature Engineering.ipynb
│   ├── 5.Train Test Split and Preparation.ipynb
│   ├── 6.Model Training Logistic Regression.ipynb
│   ├── 7.Model Training Decision Tree.ipynb
│   ├── 8.Model Training Random Forest.ipynb
│   └── 9.Model Training SVM.ipynb
│
├── 📊 data/
│   ├── raw/                               # Original datasets (6 files)
│   │   ├── Crop_recommendation.csv       # Primary dataset (2,200 rows)
│   │   ├── Crop and fertilizer dataset.csv
│   │   ├── rotation_rules.csv            # 343 rotation combinations
│   │   ├── crop-area-and-production.xlsx
│   │   └── Season_Price_Arrival_*.csv    # Market price data
│   │
│   ├── processed/                         # Cleaned & transformed (22+ files)
│   │   ├── crop_data_cleaned.csv
│   │   ├── crop_data_engineered.csv
│   │   ├── ml_ready/                     # Train/test splits, scalers
│   │   └── ...
│   │
│   ├── results/                           # Model outputs (17 files)
│   │   ├── model_comparison_all.csv
│   │   ├── *_classification_report.csv
│   │   ├── *_predictions.csv
│   │   └── *_summary.csv
│   │
│   └── visualizations/                    # Generated plots (43 files)
│       ├── 01-23: EDA visualizations
│       ├── 24-28: Logistic Regression plots
│       ├── 29-33: Decision Tree plots
│       ├── 34-39: Random Forest plots
│       └── 40-43: SVM plots
│
├── 🤖 models/                             # Trained ML Models
│   ├── logistic_regression_model.pkl     # ~8 KB
│   ├── decision_tree_model.pkl           # ~25 KB
│   ├── random_forest_model.pkl           # ~3 MB
│   ├── svm_model.pkl                     # ~315 KB
│   ├── label_encoder.pkl
│   ├── scaler_standard.pkl
│   └── scaler_minmax.pkl
│
├── 🌐 webapp/                             # Flask Application (planned)
│   ├── templates/                         # HTML templates (empty)
│   └── static/                            # CSS, JS, images (empty)
│       ├── css/
│       ├── js/
│       └── images/
│
├── 📄 docs/                               # Documentation (planned)
│   ├── presentation/
│   ├── project_report/
│   └── user_manual/
│
├── 🧪 tests/                              # Unit tests (planned)
├── ⚙️ config/                             # Configuration (planned)
│
├── 📝 README.md                           # This file
├── 📦 requirements.txt                    # Python dependencies
├── 📓 app.ipynb                           # Additional notebook
└── 📄 LICENSE                             # MIT License
```

---

## 📊 Datasets

### Primary Dataset: Crop Recommendation
| Attribute | Details |
|-----------|---------|
| **Source** | [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| **Size** | 2,200 rows × 8 columns |
| **Features** | N, P, K, temperature, humidity, pH, rainfall |
| **Target** | label (22 crop types) |

### Feature Descriptions

| Feature | Description | Range |
|---------|-------------|-------|
| **N** | Nitrogen content in soil (kg/ha) | 0-140 |
| **P** | Phosphorus content in soil (kg/ha) | 5-145 |
| **K** | Potassium content in soil (kg/ha) | 5-205 |
| **temperature** | Average temperature (°C) | 8.8-43.7 |
| **humidity** | Relative humidity (%) | 14.3-100 |
| **ph** | Soil pH value | 3.5-10 |
| **rainfall** | Annual rainfall (mm) | 20.2-298.6 |

### Supported Crops (22 Classes)

```
apple, banana, blackgram, chickpea, coconut, coffee, cotton, 
grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, 
mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, 
rice, watermelon
```

### Additional Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| `rotation_rules.csv` | 343 rows | Crop rotation compatibility rules |
| `Crop and fertilizer dataset.csv` | 377 KB | Extended fertilizer information |
| `crop-area-and-production.xlsx` | 40 KB | Historical production data |
| `Season_Price_Arrival_*.csv` | ~2 KB each | Market price data |

---

## 🏆 Model Performance

### Model Comparison Summary

| Model | Test Accuracy | Precision | Recall | F1-Score | Overfitting Gap |
|-------|---------------|-----------|--------|----------|-----------------|
| **Random Forest** 🥇 | **99.32%** | 99.35% | 99.32% | 99.32% | 0.57% |
| **SVM (RBF)** 🥈 | 97.95% | 98.09% | 97.95% | 97.94% | 1.48% |
| **Logistic Regression** 🥉 | 97.73% | 97.93% | 97.73% | 97.71% | 1.14% |
| **Decision Tree** | 95.68% | 95.92% | 95.68% | 95.70% | 1.88% |

### Best Model: Random Forest
- **Configuration**: 100 estimators, max_depth=None
- **Accuracy**: 99.32% (test set)
- **Key Features**: pH, Potassium (K), Humidity

### SVM Model Details
- **Kernel**: RBF (Radial Basis Function)
- **C Parameter**: 100
- **Support Vectors**: 646 (36.7% of training data)
- **Accuracy**: 97.95%

### Training Performance

| Model | Training Time | Prediction Time (per sample) | Model Size |
|-------|---------------|------------------------------|------------|
| Decision Tree | 0.02s | 0.004ms | 25 KB |
| Logistic Regression | 0.20s | 0.003ms | 8 KB |
| Random Forest | 0.38s | 0.257ms | 3 MB |
| SVM | 20.88s | 0.182ms | 315 KB |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/crop-recommendation-system.git
cd crop-recommendation-system
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 📖 Usage

### Running the Notebooks

Execute notebooks in order for the complete pipeline:

```bash
# 1. Data Loading
jupyter notebook notebooks/01_Data_Collection_and_Loading.ipynb

# 2. Data Cleaning
jupyter notebook notebooks/2.Data_Cleaning_and_Preprocessing.ipynb

# 3. EDA
jupyter notebook notebooks/3.Exploratory_Data_Analysis.ipynb

# Continue with remaining notebooks...
```

### Using Trained Models

```python
import pickle
import numpy as np

# Load the best model (Random Forest)
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('models/scaler_standard.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Example prediction
# Features: N, P, K, temperature, humidity, pH, rainfall (+ engineered features)
sample_input = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])  
# Note: Full feature engineering needed for 39 features

# Predict
prediction = model.predict(sample_input)
crop_name = label_encoder.inverse_transform(prediction)
print(f"Recommended Crop: {crop_name[0]}")
```

---

## 📓 Notebooks Guide

### Phase 1: Data Preparation (Notebooks 1-5)

| # | Notebook | Description | Key Outputs |
|---|----------|-------------|-------------|
| 1 | Data Collection | Load raw datasets | `crop_data_loaded.csv` |
| 2 | Data Cleaning | Handle missing values, outliers | `crop_data_cleaned.csv` |
| 3 | EDA | 43 visualizations, statistics | `visualizations/*.png` |
| 4 | Feature Engineering | Create 39 features | `crop_data_engineered.csv` |
| 5 | Train-Test Split | Prepare ML-ready data | `ml_ready/*.npy` |

### Phase 2: Model Training (Notebooks 6-9)

| # | Notebook | Algorithm | Accuracy | Status |
|---|----------|----------|----------|--------|
| 6 | Logistic Regression | Linear classifier | 97.73% | ✅ Complete |
| 7 | Decision Tree | Tree-based classifier | 95.68% | ✅ Complete |
| 8 | Random Forest | Ensemble (bagging) | 99.32% | ✅ Complete |
| 9 | SVM | Support Vector Machine | 97.95% | ✅ Complete |

### Phase 3: Advanced Models (Planned)

| # | Notebook | Algorithm | Status |
|---|----------|----------|--------|
| 10 | XGBoost & LightGBM | Gradient boosting | ⏳ Planned |
| 11 | Stacking Ensemble | Meta-learner | ⏳ Planned |
| 12 | Model Comparison | Final selection | ⏳ Planned |
| 13 | SHAP Explainability | XAI visualizations | ⏳ Planned |
| 14 | Economic Analysis | ROI calculations | ⏳ Planned |
| 15 | Crop Rotation | Multi-season planning | ⏳ Planned |
| 16 | Final Pipeline | Deployment ready | ⏳ Planned |

---

## 📈 Visualizations Generated

### EDA Visualizations (1-23)
- Feature distributions (histograms, boxplots)
- Correlation heatmap
- Pairplot of all features
- Crop distribution analysis
- Violin plots by crop
- 3D NPK scatter plot
- Climate zone analysis

### Model Performance Visualizations (24-43)
- Confusion matrices (raw and normalized)
- Per-class performance (Precision, Recall, F1)
- Feature importance charts
- Confidence distribution plots
- Decision tree structure
- Support vectors analysis (SVM)

---

## 🗺️ Future Roadmap

### Short-term Goals
- [ ] Implement XGBoost and LightGBM models
- [ ] Create Stacking Ensemble for improved accuracy
- [ ] Add SHAP explainability module
- [ ] Build Flask web application

### Medium-term Goals
- [ ] Implement economic viability analysis
- [ ] Add crop rotation planning module
- [ ] Create REST API for predictions
- [ ] Deploy to cloud platform

### Long-term Goals
- [ ] Mobile application development
- [ ] Real-time weather data integration
- [ ] Regional customization (India-specific)
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Prathamesh Gawas**

- GitHub: [@prathameshgawas](https://github.com/prathameshgawas)

---

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the Crop Recommendation Dataset
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Government of India - Agmarknet](https://agmarknet.gov.in/) for market price data
- [ICAR](https://icar.org.in/) for crop rotation research

---

<p align="center">
  <b>⭐ Star this repository if you found it helpful! ⭐</b>
</p>
