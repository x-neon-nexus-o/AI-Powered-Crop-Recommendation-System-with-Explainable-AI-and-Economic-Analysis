
# **AI-Powered Crop Recommendation System with Explainable AI and Economic Analysis**

## **Complete Project Description**

### **Project Title**
Intelligent Crop Recommendation System with Explainability, Multi-Season Planning, and Economic Viability Analysis for Sustainable Agriculture

### **Project Overview**
This semester project develops an advanced crop recommendation system that goes beyond traditional ML predictions by incorporating:
1. **Explainable AI (XAI)** - SHAP-based explanations for farmer trust
2. **Economic Viability Analysis** - Market price integration and profit calculation
3. **Multi-Season Crop Rotation Planning** - Sustainable agriculture recommendations
4. **Ensemble ML Models** - Stacking Random Forest, XGBoost, and LightGBM
5. **Interactive Web Dashboard** - Flask-based application with visualizations

**Alignment with Research Gaps:** Addresses 5 major gaps identified in existing literature - no XAI implementation, no economic analysis, no crop rotation planning, static datasets, and lack of regional customization. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/281231/6f937956-8f5c-4e8c-9f67-398198cdd951/MAJ-2025-1221-FD_1754609106.pdf)

***

## **Technology Stack**

### **Backend (Web Application Only)**
- **Flask 3.0** - Web framework
- **Python 3.10+** - Core programming language

### **Machine Learning & Data Science** (Jupyter Notebooks - Aligned with Syllabus Units I-IV)
- **Pandas** - Data manipulation (Unit I)
- **NumPy** - Numerical operations
- **scikit-learn** - ML algorithms (Unit III, IV)
- **XGBoost** - Gradient boosting (Unit IV)
- **LightGBM** - Ensemble learning
- **SHAP** - Explainable AI

### **Data Visualization** (Unit II)
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

### **Frontend (Python Only)**
- **HTML5/CSS3/Bootstrap 5** - UI framework
- **JavaScript** - Client-side interactivity
- **Chart.js** - Dashboard visualizations

### **Deployment**
- **Pickle** - Model serialization
- **Gunicorn** - WSGI server

***

## **Comprehensive Folder Structure**

```
crop-recommendation-system/
│
├── notebooks/                                    # ALL ML WORK IN JUPYTER
│   │
│   ├── 01_Data_Collection_and_Loading.ipynb    # Unit I: Pandas basics
│   │   # - Load datasets from CSV
│   │   # - Explore data structures (Series, DataFrame)
│   │   # - Data info and basic statistics
│   │   # - Save processed data
│   │
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb # Unit I: Data preparation
│   │   # - Handle missing values (dropna, fillna)
│   │   # - Remove duplicates
│   │   # - Outlier detection (IQR method)
│   │   # - Data type conversions
│   │   # - Save cleaned data
│   │
│   ├── 03_Exploratory_Data_Analysis.ipynb       # Unit II: Visualization
│   │   # - Matplotlib plots (line, bar, scatter, histogram)
│   │   # - Seaborn plots (heatmap, boxplot, violin)
│   │   # - Distribution analysis
│   │   # - Correlation analysis
│   │   # - Save EDA insights
│   │
│   ├── 04_Feature_Engineering.ipynb             # Unit I: Data transformation
│   │   # - Create new features (NPK ratio, temp-humidity index)
│   │   # - Data aggregation (groupby)
│   │   # - Data merging and joining
│   │   # - Categorical encoding
│   │   # - Save engineered features
│   │
│   ├── 05_Statistical_Analysis.ipynb            # Unit II: Statistics
│   │   # - Descriptive statistics (mean, median, std)
│   │   # - Correlation and covariance
│   │   # - Hypothesis testing (ANOVA, t-test, chi-square)
│   │   # - Feature significance testing
│   │   # - Save statistical results
│   │
│   ├── 06_Data_Normalization_and_Splitting.ipynb # Unit I: Data preparation
│   │   # - StandardScaler for normalization
│   │   # - Train-test split (80-20)
│   │   # - Save train/test sets
│   │   # - Save scaler object
│   │
│   ├── 07_Baseline_Classification_Models.ipynb  # Unit III: Supervised Learning
│   │   # - Logistic Regression
│   │   # - k-Nearest Neighbors (KNN)
│   │   # - Naïve Bayes
│   │   # - Support Vector Machine (SVM)
│   │   # - Model evaluation (accuracy, confusion matrix)
│   │   # - Save baseline models
│   │
│   ├── 08_Decision_Tree_Classifier.ipynb        # Unit IV: Decision Trees
│   │   # - Build Decision Tree model
│   │   # - Hyperparameter tuning (max_depth, min_samples_split)
│   │   # - Feature importance visualization
│   │   # - Save Decision Tree model
│   │
│   ├── 09_Random_Forest_Classifier.ipynb        # Unit IV: Random Forests
│   │   # - Build Random Forest model
│   │   # - GridSearchCV for optimization
│   │   # - Cross-validation (5-fold)
│   │   # - Feature importance analysis
│   │   # - Save Random Forest model
│   │
│   ├── 10_XGBoost_and_LightGBM.ipynb           # Unit IV: Ensemble Methods
│   │   # - Build XGBoost model
│   │   # - Build LightGBM model
│   │   # - Hyperparameter tuning
│   │   # - Performance comparison
│   │   # - Save XGBoost and LightGBM models
│   │
│   ├── 11_Stacking_Ensemble_Model.ipynb         # Unit IV: Advanced Ensemble
│   │   # - Create stacking classifier
│   │   # - Base models: RF, XGBoost, LightGBM
│   │   # - Meta-learner: Logistic Regression
│   │   # - Final model evaluation
│   │   # - Save final ensemble model
│   │
│   ├── 12_Model_Comparison_and_Selection.ipynb  # Unit III: Model Evaluation
│   │   # - Compare all models (accuracy, precision, recall, F1)
│   │   # - ROC-AUC curves
│   │   # - Confusion matrix comparison
│   │   # - Select best model
│   │   # - Save comparison report
│   │
│   ├── 13_Explainable_AI_with_SHAP.ipynb       # XAI Implementation
│   │   # - Load best model
│   │   # - Generate SHAP values
│   │   # - SHAP waterfall plots
│   │   # - SHAP summary plots
│   │   # - SHAP force plots
│   │   # - Feature contribution analysis
│   │   # - Save SHAP explainer
│   │
│   ├── 14_Economic_Viability_Analysis.ipynb     # Economic Module
│   │   # - Load crop price dataset
│   │   # - Build profit calculator
│   │   # - ROI calculation: (Yield × Price - Costs) / Costs
│   │   # - Risk assessment scoring
│   │   # - Cost-benefit analysis
│   │   # - Save economic analysis functions
│   │
│   ├── 15_Crop_Rotation_Planning.ipynb          # Rotation Module
│   │   # - Load rotation rules dataset
│   │   # - Build rule-based engine
│   │   # - Multi-season planning algorithm
│   │   # - Soil health scoring (N-P-K depletion/addition)
│   │   # - Sustainability metrics
│   │   # - Save rotation planning logic
│   │
│   └── 16_Final_Pipeline_and_Export.ipynb       # Integration
│       # - Load all components
│       # - Create prediction pipeline
│       # - Test end-to-end workflow
│       # - Export all models and utilities
│       # - Generate deployment-ready files
│
├── data/                               # All Datasets
│   ├── raw/
│   │   ├── crop_recommendation.csv    # Base dataset (Kaggle)
│   │   ├── crop_prices.csv            # Market prices dataset
│   │   ├── regional_varieties.csv     # Maharashtra-specific varieties
│   │   ├── rotation_rules.csv         # Crop rotation knowledge base
│   │   └── soil_nutrient_standards.csv # NPK requirement ranges
│   │
│   ├── processed/
│   │   ├── cleaned_data.csv           # After preprocessing
│   │   ├── engineered_features.csv    # Feature engineering output
│   │   ├── normalized_data.csv        # After scaling
│   │   └── train_test_split/
│   │       ├── X_train.csv
│   │       ├── X_test.csv
│   │       ├── y_train.csv
│   │       └── y_test.csv
│   │
│   └── results/
│       ├── model_comparison.csv       # All model accuracies
│       ├── feature_importance.csv     # Feature rankings
│       ├── shap_values.csv           # SHAP analysis results
│       └── economic_analysis.csv      # ROI calculations
│
├── models/                             # Saved Models (from notebooks)
│   ├── baseline_models/
│   │   ├── logistic_regression.pkl
│   │   ├── knn_classifier.pkl
│   │   ├── naive_bayes.pkl
│   │   └── svm_classifier.pkl
│   │
│   ├── tree_models/
│   │   ├── decision_tree.pkl
│   │   └── random_forest.pkl
│   │
│   ├── boosting_models/
│   │   ├── xgboost_model.pkl
│   │   └── lightgbm_model.pkl
│   │
│   ├── ensemble/
│   │   └── stacking_ensemble.pkl      # FINAL BEST MODEL
│   │
│   ├── preprocessing/
│   │   ├── scaler.pkl                 # StandardScaler
│   │   └── label_encoder.pkl          # Crop name encoding
│   │
│   ├── explainability/
│   │   └── shap_explainer.pkl         # SHAP TreeExplainer
│   │
│   └── metadata/
│       ├── model_metrics.json         # Performance comparison
│       ├── feature_names.json         # Feature list
│       └── crop_labels.json           # Crop name mappings
│
├── webapp/                             # FLASK APPLICATION (ONLY .py FILES)
│   │
│   ├── app.py                         # Main Flask application
│   │   # - Flask app initialization
│   │   # - Route definitions
│   │   # - Model loading
│   │   # - Run server
│   │
│   ├── utils.py                       # Utility functions
│   │   # - load_all_models()
│   │   # - validate_input()
│   │   # - prepare_features()
│   │   # - format_output()
│   │
│   ├── prediction.py                  # Prediction logic
│   │   # - predict_crop()
│   │   # - get_top_n_crops()
│   │   # - confidence_scores()
│   │
│   ├── explainability.py              # XAI logic
│   │   # - generate_shap_explanation()
│   │   # - create_shap_plots()
│   │   # - feature_contribution_text()
│   │
│   ├── economic.py                    # Economic analysis
│   │   # - calculate_roi()
│   │   # - get_market_prices()
│   │   # - cost_benefit_analysis()
│   │   # - rank_by_profitability()
│   │
│   ├── rotation.py                    # Rotation planning
│   │   # - get_rotation_suggestions()
│   │   # - plan_multiseason()
│   │   # - calculate_soil_impact()
│   │   # - sustainability_score()
│   │
│   ├── templates/                     # HTML templates
│   │   ├── base.html                 # Base template with navbar
│   │   ├── index.html                # Landing page
│   │   ├── input_form.html           # Data input form
│   │   ├── results.html              # Prediction results
│   │   ├── explanation.html          # SHAP visualizations
│   │   ├── economic_dashboard.html   # ROI analysis
│   │   ├── rotation_plan.html        # Multi-season plan
│   │   └── comparison.html           # Compare multiple crops
│   │
│   └── static/                        # CSS, JS, Images
│       ├── css/
│       │   ├── bootstrap.min.css
│       │   └── custom_styles.css
│       ├── js/
│       │   ├── chart.min.js
│       │   ├── plotly.min.js
│       │   ├── dashboard.js          # Chart configurations
│       │   └── form_validation.js
│       └── images/
│           ├── logo.png
│           ├── crop_icons/
│           └── shap_plots/           # Generated SHAP images
│
├── docs/                               # Documentation
│   ├── project_report/
│   │   ├── 01_introduction.md
│   │   ├── 02_literature_review.md
│   │   ├── 03_methodology.md
│   │   ├── 04_results.md
│   │   ├── 05_conclusion.md
│   │   └── final_report.pdf
│   │
│   ├── presentation/
│   │   └── project_presentation.pptx
│   │
│   ├── user_manual/
│   │   └── user_guide.pdf
│   │
│   └── api_documentation.md           # Web API docs
│
├── tests/                              # Unit Tests (Optional)
│   ├── test_prediction.py
│   ├── test_economic.py
│   └── test_rotation.py
│
├── config/                             # Configuration
│   └── config.py                      # Flask config, paths
│
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── .gitignore
└── LICENSE
```

***

## **Required Datasets with Sources**

### **1. Base Crop Recommendation Dataset** ⭐ PRIMARY
**Source:** Kaggle - Crop Recommendation Dataset
```
Direct Link: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
Features: N, P, K, temperature, humidity, pH, rainfall, label
Crops: 22 (Rice, Wheat, Maize, Cotton, Jute, etc.)
Size: 2,200 rows × 8 columns
Format: CSV
File: data/raw/crop_recommendation.csv

Download: Click "Download" on Kaggle page
```

### **2. Market Price Dataset**
**Source:** Government of India - Agmarknet Portal
```
Option 1 (Recommended): Manual compilation
- Visit: https://agmarknet.gov.in/
- Navigate: Price & Arrivals → Commodity Wise Daily Prices
- Download: Monthly reports for major crops
- Compile: Aggregate into single CSV

Option 2: data.gov.in
- URL: https://data.gov.in/
- Search: "agricultural prices" or "APMC prices"
- Download: Available CSV files

Required Columns:
- Crop, State, Date, Modal_Price, Min_Price, Max_Price, Unit

File: data/raw/crop_prices.csv
Expected Size: 500-1000 rows (monthly data for 1-2 years)
```

### **3. Crop Rotation Knowledge Base**
**Source:** Create manually from agricultural research
```
Reference: Indian Council of Agricultural Research (ICAR)
URL: https://icar.org.in/

Create CSV with columns:
Crop1, Crop2, Season1, Season2, Compatibility_Score, Soil_Impact

Example entries:
Rice,Wheat,Kharif,Rabi,0.9,Neutral
Wheat,Mung Bean,Rabi,Kharif,0.85,Positive
Cotton,Chickpea,Kharif,Rabi,0.75,Neutral
Maize,Groundnut,Kharif,Kharif,0.6,Negative

File: data/raw/rotation_rules.csv
Size: 50-100 rotation combinations
```



### **Quick Dataset Download Summary**

| Dataset | Source | Action | Priority |
|---------|--------|--------|----------|
| Crop Recommendation | Kaggle | Download directly | **MUST HAVE** |
| Market Prices | Agmarknet | Manual compilation | **SHOULD HAVE** |
| Rotation Rules | Self-created | Create from research | **SHOULD HAVE** |

***

## **Jupyter Notebooks Detailed Breakdown**

### **Notebook 01: Data Collection and Loading** (Week 1)
```python
# Content Overview:
- Import pandas, numpy
- Load crop_recommendation.csv
- Explore DataFrame structure (.info(), .describe(), .head())
- Check data types and missing values
- Load additional datasets (prices, rotation rules)
- Save to processed/ folder

# Key Learning: Unit I - Pandas basics, data loading
```

### **Notebook 02: Data Cleaning and Preprocessing** (Week 1-2)
```python
# Content Overview:
- Check for missing values (isnull().sum())
- Handle missing data (dropna(), fillna() with mean/median)
- Remove duplicates (drop_duplicates())
- Outlier detection using IQR method
- Box plots for outlier visualization
- Data type conversions
- Save cleaned_data.csv

# Key Learning: Unit I - Data cleansing, handling missing data
```

### **Notebook 03: Exploratory Data Analysis** (Week 2)
```python
# Content Overview:
- Matplotlib visualizations:
  * Histograms for feature distributions
  * Scatter plots (N vs P, Temp vs Humidity)
  * Bar plots for crop frequency
- Seaborn visualizations:
  * Correlation heatmap
  * Box plots for outliers
  * Pair plots for feature relationships
  * Violin plots for distributions
- Statistical summaries (mean, median, std)
- Save EDA insights

# Key Learning: Unit II - Matplotlib, Seaborn plotting
```

### **Notebook 04: Feature Engineering** (Week 3)
```python
# Content Overview:
- Create new features:
  * NPK_ratio = N / (P + K)
  * Temp_Humidity_Index = Temperature / Humidity
  * Nutrient_Balance = (N + P + K) / 3
- Data aggregation (groupby crop → mean values)
- Merge datasets (crop data + prices)
- Categorical encoding (LabelEncoder for crops)
- Save engineered_features.csv

# Key Learning: Unit I - Data transformation, aggregation, merging
```

### **Notebook 05: Statistical Analysis** (Week 3)
```python
# Content Overview:
- Descriptive statistics (describe())
- Correlation analysis (corr())
- Covariance matrix
- ANOVA test for feature significance
- Chi-square test for categorical relationships
- T-test for group comparisons
- Feature selection based on p-values
- Save statistical_results.csv

# Key Learning: Unit II - SciPy statistics, hypothesis testing
```

### **Notebook 06: Data Normalization and Splitting** (Week 3)
```python
# Content Overview:
- Import StandardScaler from sklearn
- Normalize features (fit_transform)
- Train-test split (80-20, stratified)
- Save X_train, X_test, y_train, y_test
- Save scaler object (pickle)
- Verify split balance

# Key Learning: Unit I - Data preparation for ML
```

### **Notebook 07: Baseline Classification Models** (Week 4-5)
```python
# Content Overview:
- Logistic Regression
- k-Nearest Neighbors (k=5)
- Naïve Bayes (GaussianNB)
- Support Vector Machine (kernel='rbf')
- Model training and prediction
- Accuracy, confusion matrix, classification report
- Save all baseline models
- Create comparison table

# Key Learning: Unit III - Supervised classification algorithms
```

### **Notebook 08: Decision Tree Classifier** (Week 5)
```python
# Content Overview:
- Build Decision Tree (max_depth=10)
- Hyperparameter tuning (GridSearchCV)
- Feature importance extraction
- Visualize tree structure
- Plot feature importance bar chart
- Save decision_tree.pkl

# Key Learning: Unit IV - Decision Trees
```

### **Notebook 09: Random Forest Classifier** (Week 6)
```python
# Content Overview:
- Build Random Forest (n_estimators=100)
- GridSearchCV for optimization
- 5-fold cross-validation
- Feature importance analysis
- Out-of-bag score
- Save random_forest.pkl

# Key Learning: Unit IV - Random Forests, ensemble learning
```

### **Notebook 10: XGBoost and LightGBM** (Week 6)
```python
# Content Overview:
- Build XGBoost classifier
- Build LightGBM classifier
- Hyperparameter tuning for both
- Performance comparison
- ROC-AUC curves
- Save xgboost_model.pkl, lightgbm_model.pkl

# Key Learning: Unit IV - Advanced ensemble methods
```

### **Notebook 11: Stacking Ensemble Model** (Week 7)
```python
# Content Overview:
- Create StackingClassifier
- Base estimators: Random Forest, XGBoost, LightGBM
- Meta-learner: Logistic Regression
- Train stacking model
- Final evaluation (target: >98% accuracy)
- Save stacking_ensemble.pkl (FINAL MODEL)

# Key Learning: Unit IV - Stacking ensemble
```

### **Notebook 12: Model Comparison and Selection** (Week 7)
```python
# Content Overview:
- Load all saved models
- Compare accuracy, precision, recall, F1-score
- ROC-AUC curves for all models
- Confusion matrices side-by-side
- Select best model (Stacking Ensemble)
- Create model_comparison.csv
- Generate comparison visualizations

# Key Learning: Unit III - Model evaluation
```

### **Notebook 13: Explainable AI with SHAP** (Week 8-9)
```python
# Content Overview:
- Install SHAP library
- Load best model (Stacking Ensemble)
- Create TreeExplainer
- Generate SHAP values for test set
- Create visualizations:
  * SHAP waterfall plot (single prediction)
  * SHAP summary plot (all features)
  * SHAP force plot (interactive)
  * SHAP dependence plots
- Interpret feature contributions
- Save shap_explainer.pkl

# Key Learning: Beyond syllabus - Explainable AI
```

### **Notebook 14: Economic Viability Analysis** (Week 9-10)
```python
# Content Overview:
- Load crop_prices.csv
- Build profit calculator function
- Calculate ROI: (Expected_Yield × Market_Price - Input_Costs) / Input_Costs
- Risk scoring (price volatility analysis)
- Cost-benefit analysis for each crop
- Rank crops by profitability
- Save economic_analysis.csv

# Key Learning: Beyond syllabus - Economic modeling
```

### **Notebook 15: Crop Rotation Planning** (Week 10-11)
```python
# Content Overview:
- Load rotation_rules.csv
- Build rule-based recommendation engine
- Multi-season planning algorithm:
  * Season 1: Recommended crop
  * Season 2: Compatible rotation crop
  * Season 3: Soil recovery crop
- Soil nutrient depletion/addition tracking
- Sustainability score calculation
- Save rotation planning functions

# Key Learning: Beyond syllabus - Rule-based AI
```

### **Notebook 16: Final Pipeline and Export** (Week 11)
```python
# Content Overview:
- Load all components (model, scaler, explainer)
- Create end-to-end prediction pipeline
- Test with sample inputs
- Validate all outputs:
  * Crop prediction
  * SHAP explanation
  * Economic analysis
  * Rotation plan
- Export deployment-ready files
- Generate metadata (feature names, crop labels)

# Key Learning: Integration and deployment preparation
```

***

## **Flask Web Application Structure (.py files only)**

### **app.py** (Main Application)
```python
"""
Main Flask Application
- Initialize Flask app
- Define all routes
- Load models on startup
- Run development server
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from utils import load_all_models, validate_input
from prediction import predict_crop
from explainability import generate_shap_explanation
from economic import calculate_roi
from rotation import get_rotation_suggestions

app = Flask(__name__)

# Routes:
# @app.route('/')           - Home page
# @app.route('/predict')    - Crop prediction
# @app.route('/explain')    - SHAP explanations
# @app.route('/economic')   - Economic analysis
# @app.route('/rotation')   - Rotation planning
# @app.route('/compare')    - Compare crops
```

### **utils.py** (Utility Functions)
```python
"""
Utility Functions
- load_all_models(): Load pickle files
- validate_input(): Check input ranges
- prepare_features(): Scale and transform
- format_output(): Format predictions
"""
```

### **prediction.py** (Prediction Logic)
```python
"""
Crop Prediction Module
- predict_crop(): Main prediction function
- get_top_n_crops(): Return ranked suggestions
- confidence_scores(): Calculate probabilities
"""
```

### **explainability.py** (XAI Logic)
```python
"""
Explainability Module
- generate_shap_explanation(): Create SHAP values
- create_shap_plots(): Generate visualizations
- feature_contribution_text(): Human-readable explanations
"""
```

### **economic.py** (Economic Analysis)
```python
"""
Economic Analysis Module
- calculate_roi(): ROI calculation
- get_market_prices(): Fetch current prices
- cost_benefit_analysis(): Compare costs vs profits
- rank_by_profitability(): Sort crops by profit
"""
```

### **rotation.py** (Rotation Planning)
```python
"""
Crop Rotation Module
- get_rotation_suggestions(): Load rotation rules
- plan_multiseason(): 3-season planning
- calculate_soil_impact(): Nutrient tracking
- sustainability_score(): Environmental rating
"""
```

***

## **Semester-Friendly Implementation Timeline (14 Weeks)**

### **Phase 1: Jupyter Notebooks - Data Science (Weeks 1-7)**

| Week | Notebooks | Syllabus Unit | Deliverable |
|------|-----------|---------------|-------------|
| 1 | 01, 02 | Unit I | Cleaned dataset |
| 2 | 03 | Unit II | EDA report (10+ visualizations) |
| 3 | 04, 05, 06 | Unit I, II | Engineered features, statistical analysis |
| 4 | 07 | Unit III | Baseline models (Logistic, KNN, SVM, NB) |
| 5 | 08 | Unit IV | Decision Tree model |
| 6 | 09, 10 | Unit IV | Random Forest, XGBoost, LightGBM |
| 7 | 11, 12 | Unit IV | Stacking Ensemble (BEST MODEL) |

### **Phase 2: Jupyter Notebooks - Advanced Features (Weeks 8-11)**

| Week | Notebooks | Focus | Deliverable |
|------|-----------|-------|-------------|
| 8-9 | 13 | XAI | SHAP explainer + visualizations |
| 9-10 | 14 | Economic | ROI calculator + price analysis |
| 10-11 | 15 | Rotation | Multi-season planner |
| 11 | 16 | Integration | Final pipeline + exported models |

### **Phase 3: Flask Web Application (Weeks 12-13)**

| Week | Files | Tasks | Deliverable |
|------|-------|-------|-------------|
| 12 | app.py, utils.py, prediction.py | Backend development, routes, model integration | Working Flask API |
| 13 | explainability.py, economic.py, rotation.py, templates/, static/ | Frontend, dashboards, visualizations | Complete web app |

### **Phase 4: Documentation & Testing (Week 14)**

| Week | Tasks | Deliverable |
|------|-------|-------------|
| 14 | Testing, README, report writing, presentation | Final submission package |

***

## **Installation & Setup**

### **requirements.txt**
```txt
# Core
flask==3.0.0
gunicorn==21.2.0

# Jupyter
jupyter==1.0.0
notebook==7.0.6
ipywidgets==8.1.1

# Data Science (Unit I)
pandas==2.1.0
numpy==1.24.3

# Visualization (Unit II)
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1

# Statistics (Unit II)
scipy==1.11.2

# Machine Learning (Unit III, IV)
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0

# Explainability
shap==0.42.1

# Utilities
joblib==1.3.2
openpyxl==3.1.2
```

### **Quick Start Commands**
```bash
# Create project directory
mkdir crop-recommendation-system
cd crop-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter for ML work
jupyter notebook

# After completing all notebooks, run Flask app
cd webapp
python app.py
```

***

## **Project Deliverables**

### **ML Deliverables (Jupyter Notebooks)**
1. ✅ 16 Jupyter Notebooks covering all units
2. ✅ Trained models (9 models + 1 final ensemble)
3. ✅ SHAP explainer and visualizations
4. ✅ Economic analysis module
5. ✅ Crop rotation planner
6. ✅ Model comparison report

### **Web Application Deliverables (Flask .py files)**
7. ✅ Flask web application (6 Python files)
8. ✅ Interactive dashboards (5 HTML templates)
9. ✅ Responsive UI (Bootstrap 5)
10. ✅ API endpoints for all features

### **Documentation**
11. ✅ Comprehensive README
12. ✅ Project report (30-40 pages)
13. ✅ Presentation (15-20 slides)
14. ✅ Demo video (5-7 minutes)

***

## **Innovation Highlights**

1. ✅ **Notebook-Driven Development** - Pure Jupyter for ML (semester-friendly)
2. ✅ **XAI Integration** - First crop system with SHAP
3. ✅ **Economic Analysis** - ROI-based recommendations
4. ✅ **Multi-Season Planning** - Sustainability focus
5. ✅ **Stacking Ensemble** - Advanced ML beyond papers
6. ✅ **Clean Separation** - Notebooks (.ipynb) for ML, Python (.py) for web only

