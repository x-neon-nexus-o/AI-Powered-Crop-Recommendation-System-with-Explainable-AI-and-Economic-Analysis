
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

**Alignment with Research Gaps:** Addresses 5 major gaps identified in existing literature - no XAI implementation, no economic analysis, no crop rotation planning, static datasets, and lack of regional customization.

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
├── 📓 notebooks/                                    # ALL ML WORK - JUPYTER NOTEBOOKS
│   │
│   ├── 01_Data_Collection_and_Loading.ipynb       # Week 1 | Unit I
│   │   # Tasks:
│   │   # - Import pandas, numpy
│   │   # - Load crop_recommendation.csv from Kaggle
│   │   # - Load crop_prices.csv from Agmarknet
│   │   # - Initial data exploration (.info(), .describe(), .head())
│   │   # - Check for missing values and duplicates
│   │   # - Data type verification
│   │   # - Save to processed/ folder
│   │
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb   # Week 1-2 | Unit I
│   │   # Tasks:
│   │   # - Detect missing values (isnull().sum())
│   │   # - Handle missing data (dropna(), fillna() with mean/median)
│   │   # - Remove duplicate rows (drop_duplicates())
│   │   # - Outlier detection using IQR method
│   │   # - Box plots for outlier visualization
│   │   # - Data type conversions
│   │   # - Save cleaned_data.csv
│   │
│   ├── 03_Exploratory_Data_Analysis.ipynb         # Week 2 | Unit II
│   │   # Tasks:
│   │   # - Matplotlib visualizations (10+ plots):
│   │   #   * Histograms for N, P, K distributions
│   │   #   * Scatter plots (N vs P, Temperature vs Humidity)
│   │   #   * Bar plots for crop frequency
│   │   #   * Pie charts for crop categories
│   │   # - Seaborn visualizations:
│   │   #   * Correlation heatmap (all features)
│   │   #   * Box plots for outlier detection
│   │   #   * Pair plots for feature relationships
│   │   #   * Violin plots for distributions by crop
│   │   # - Statistical summaries (mean, median, std, variance)
│   │   # - Save EDA insights as images
│   │
│   ├── 04_Feature_Engineering.ipynb               # Week 3 | Unit I
│   │   # Tasks:
│   │   # - Create derived features:
│   │   #   * NPK_ratio = N / (P + K + 1)
│   │   #   * Temp_Humidity_Index = Temperature / Humidity
│   │   #   * Nutrient_Balance = (N + P + K) / 3
│   │   #   * pH_Category (Acidic, Neutral, Alkaline)
│   │   # - Data aggregation (groupby crop → mean, std)
│   │   # - Merge crop_prices with crop data
│   │   # - Categorical encoding (LabelEncoder for crops)
│   │   # - Save engineered_features.csv
│   │
│   ├── 05_Statistical_Analysis.ipynb              # Week 3 | Unit II
│   │   # Tasks:
│   │   # - Descriptive statistics (describe())
│   │   # - Pearson correlation analysis (corr())
│   │   # - Covariance matrix
│   │   # - ANOVA test for feature significance (scipy.stats.f_oneway)
│   │   # - Chi-square test for categorical relationships
│   │   # - T-test for comparing crop groups
│   │   # - Feature selection based on p-values (p < 0.05)
│   │   # - Save statistical_results.csv
│   │
│   ├── 06_Data_Normalization_and_Splitting.ipynb  # Week 3 | Unit I
│   │   # Tasks:
│   │   # - Import StandardScaler from sklearn.preprocessing
│   │   # - Normalize features (fit_transform on training data)
│   │   # - Train-test split (80-20, stratified by crop)
│   │   # - Save X_train, X_test, y_train, y_test as CSV
│   │   # - Save scaler object (pickle)
│   │   # - Verify class distribution balance
│   │
│   ├── 07_Baseline_Classification_Models.ipynb    # Week 4-5 | Unit III
│   │   # Tasks:
│   │   # - Logistic Regression (multi-class)
│   │   # - k-Nearest Neighbors (k=5, 7, 9 comparison)
│   │   # - Naïve Bayes (GaussianNB)
│   │   # - Support Vector Machine (kernel='rbf', 'linear')
│   │   # - Model training and prediction
│   │   # - Evaluation metrics:
│   │   #   * Accuracy score
│   │   #   * Confusion matrix
│   │   #   * Classification report (precision, recall, F1)
│   │   # - Save all baseline models as .pkl
│   │   # - Create comparison table (model_comparison.csv)
│   │
│   ├── 08_Decision_Tree_Classifier.ipynb          # Week 5 | Unit IV
│   │   # Tasks:
│   │   # - Build Decision Tree (max_depth=10, 15, 20)
│   │   # - Hyperparameter tuning with GridSearchCV
│   │   #   * max_depth, min_samples_split, min_samples_leaf
│   │   # - Feature importance extraction (.feature_importances_)
│   │   # - Tree visualization (plot_tree from sklearn)
│   │   # - Feature importance bar chart
│   │   # - Save decision_tree.pkl
│   │
│   ├── 09_Random_Forest_Classifier.ipynb          # Week 6 | Unit IV
│   │   # Tasks:
│   │   # - Build Random Forest (n_estimators=100, 200, 300)
│   │   # - GridSearchCV for optimization:
│   │   #   * n_estimators, max_depth, min_samples_split
│   │   # - 5-fold cross-validation
│   │   # - Feature importance analysis (top 10 features)
│   │   # - Out-of-bag (OOB) score
│   │   # - Partial dependence plots
│   │   # - Save random_forest.pkl
│   │
│   ├── 10_XGBoost_and_LightGBM.ipynb             # Week 6 | Unit IV
│   │   # Tasks:
│   │   # - Build XGBoost classifier (xgb.XGBClassifier)
│   │   #   * learning_rate=0.1, max_depth=6, n_estimators=100
│   │   # - Build LightGBM classifier (lgb.LGBMClassifier)
│   │   #   * num_leaves=31, learning_rate=0.1
│   │   # - Hyperparameter tuning for both
│   │   # - Performance comparison (accuracy, training time)
│   │   # - ROC-AUC curves (multi-class)
│   │   # - Feature importance comparison
│   │   # - Save xgboost_model.pkl, lightgbm_model.pkl
│   │
│   ├── 11_Stacking_Ensemble_Model.ipynb           # Week 7 | Unit IV
│   │   # Tasks:
│   │   # - Create StackingClassifier (from sklearn.ensemble)
│   │   # - Base estimators (Level 0):
│   │   #   * Random Forest (n_estimators=200)
│   │   #   * XGBoost (best params from Notebook 10)
│   │   #   * LightGBM (best params from Notebook 10)
│   │   # - Meta-learner (Level 1):
│   │   #   * Logistic Regression (C=1.0)
│   │   # - Train stacking model
│   │   # - Final evaluation (target: >98% accuracy)
│   │   # - Confusion matrix heatmap
│   │   # - Save stacking_ensemble.pkl (FINAL BEST MODEL)
│   │
│   ├── 12_Model_Comparison_and_Selection.ipynb    # Week 7 | Unit III
│   │   # Tasks:
│   │   # - Load all saved models (9 models total)
│   │   # - Compare performance metrics:
│   │   #   * Accuracy, Precision, Recall, F1-score
│   │   #   * Training time, Prediction time
│   │   #   * Model size (KB)
│   │   # - ROC-AUC curves (all models on same plot)
│   │   # - Confusion matrix comparison (3x3 subplot)
│   │   # - Select best model (Stacking Ensemble)
│   │   # - Create detailed comparison table
│   │   # - Save model_comparison.csv, comparison_plots.png
│   │
│   ├── 13_Explainable_AI_with_SHAP.ipynb         # Week 8-9 | XAI
│   │   # Tasks:
│   │   # - Install SHAP: pip install shap
│   │   # - Load best model (Stacking Ensemble)
│   │   # - Create TreeExplainer (shap.TreeExplainer)
│   │   # - Generate SHAP values for test set (100 samples)
│   │   # - Create visualizations:
│   │   #   * SHAP waterfall plot (single prediction explanation)
│   │   #   * SHAP summary plot (global feature importance)
│   │   #   * SHAP force plot (interactive HTML)
│   │   #   * SHAP dependence plots (N, P, K, Rainfall)
│   │   #   * SHAP bar plot (mean absolute SHAP values)
│   │   # - Interpret feature contributions
│   │   # - Generate explanation text (top 3 features per crop)
│   │   # - Save shap_explainer.pkl, shap_values.csv
│   │
│   ├── 14_Economic_Viability_Analysis.ipynb       # Week 9-10 | Economic
│   │   # Tasks:
│   │   # - Load crop_prices.csv
│   │   # - Build profit calculator function:
│   │   #   def calculate_roi(crop, season):
│   │   #       price = get_market_price(crop, season)
│   │   #       yield = estimate_yield(crop)  # quintals/acre
│   │   #       revenue = price * yield
│   │   #       costs = seed + fertilizer + labor
│   │   #       profit = revenue - costs
│   │   #       roi = (profit / costs) * 100
│   │   #       return roi
│   │   # - Calculate ROI for all 22 crops
│   │   # - Price volatility analysis (std, CV)
│   │   # - Risk scoring (high/medium/low based on volatility)
│   │   # - Cost-benefit analysis table
│   │   # - Rank crops by profitability
│   │   # - Create economic dashboard data
│   │   # - Save economic_analysis.csv
│   │
│   ├── 15_Crop_Rotation_Planning.ipynb            # Week 10-11 | Rotation (OPTIONAL)
│   │   # Tasks:
│   │   # - Load rotation_rules.csv (create if not exists)
│   │   # - Build rule-based recommendation engine:
│   │   #   def get_rotation_plan(current_crop, season):
│   │   #       compatible_crops = filter_by_compatibility(current_crop)
│   │   #       next_season_crop = select_best_rotation(compatible_crops)
│   │   #       return [Season1: current, Season2: next, Season3: recovery]
│   │   # - Soil nutrient tracking algorithm:
│   │   #   * Legumes (Chickpea, Mung Bean) → Add nitrogen
│   │   #   * Cereals (Rice, Wheat) → Deplete nitrogen
│   │   #   * Oilseeds (Groundnut) → Neutral
│   │   # - Sustainability scoring (0-100):
│   │   #   * 80-100: Excellent rotation
│   │   #   * 60-79: Good rotation
│   │   #   * <60: Poor rotation
│   │   # - Multi-season planning (3 seasons)
│   │   # - Save rotation functions for Flask integration
│   │
│   └── 16_Final_Pipeline_and_Export.ipynb         # Week 11 | Integration
│       # Tasks:
│       # - Load all components:
│       #   * Best model (stacking_ensemble.pkl)
│       #   * Scaler (scaler.pkl)
│       #   * Label encoder (label_encoder.pkl)
│       #   * SHAP explainer (shap_explainer.pkl)
│       # - Create end-to-end prediction pipeline:
│       #   def predict_crop_pipeline(N, P, K, temp, humidity, pH, rainfall):
│       #       1. Validate inputs
│       #       2. Scale features
│       #       3. Predict crop (top 3 probabilities)
│       #       4. Generate SHAP explanation
│       #       5. Calculate economic viability
│       #       6. Suggest rotation plan
│       #       return complete_recommendation
│       # - Test with 10 sample inputs
│       # - Validate all outputs
│       # - Export deployment files:
│       #   * All .pkl models
│       #   * feature_names.json
│       #   * crop_labels.json
│       #   * model_metadata.json
│       # - Generate deployment checklist
│
├── 📊 data/                                        # ALL DATASETS
│   │
│   ├── raw/                                        # Original datasets
│   │   ├── crop_recommendation.csv                # ✅ Kaggle - 2200 rows, 8 cols
│   │   │   # Columns: N, P, K, temperature, humidity, ph, rainfall, label
│   │   │   # Crops: 22 (Rice, Wheat, Maize, Cotton, Chickpea, etc.)
│   │   │
│   │   ├── crop_prices.csv                        # ✅ Agmarknet - 100-150 rows
│   │   │   # Columns: Crop, Season, Year, State, Modal_Price, Min_Price, Max_Price
│   │   │   # Seasons: Kharif, Rabi, Zaid
│   │   │   # Years: 2023, 2024, 2025
│   │   │
│   │   └── rotation_rules.csv                     # ⚠️ OPTIONAL - 50-80 rows
│   │       # Columns: Crop1, Crop2, Season1, Season2, Compatibility_Score, Soil_Impact
│   │       # Example: Rice,Wheat,Kharif,Rabi,0.9,Neutral
│   │
│   ├── processed/                                  # Cleaned & transformed
│   │   ├── cleaned_data.csv                       # From Notebook 02
│   │   ├── engineered_features.csv                # From Notebook 04
│   │   ├── normalized_data.csv                    # From Notebook 06
│   │   │
│   │   └── train_test_split/                      # From Notebook 06
│   │       ├── X_train.csv                        # Features (1760 rows)
│   │       ├── X_test.csv                         # Features (440 rows)
│   │       ├── y_train.csv                        # Labels (1760 rows)
│   │       └── y_test.csv                         # Labels (440 rows)
│   │
│   └── results/                                    # Analysis outputs
│       ├── model_comparison.csv                   # From Notebook 12
│       ├── feature_importance.csv                 # From Notebook 09
│       ├── shap_values.csv                        # From Notebook 13
│       ├── economic_analysis.csv                  # From Notebook 14
│       └── statistical_tests.csv                  # From Notebook 05
│
├── 🤖 models/                                      # SAVED ML MODELS
│   │
│   ├── baseline_models/                           # From Notebook 07
│   │   ├── logistic_regression.pkl               # ~50 KB
│   │   ├── knn_classifier.pkl                    # ~100 KB
│   │   ├── naive_bayes.pkl                       # ~20 KB
│   │   └── svm_classifier.pkl                    # ~80 KB
│   │
│   ├── tree_models/                               # From Notebooks 08-09
│   │   ├── decision_tree.pkl                     # ~30 KB
│   │   └── random_forest.pkl                     # ~500 KB
│   │
│   ├── boosting_models/                           # From Notebook 10
│   │   ├── xgboost_model.pkl                     # ~300 KB
│   │   └── lightgbm_model.pkl                    # ~250 KB
│   │
│   ├── ensemble/                                  # From Notebook 11
│   │   └── stacking_ensemble.pkl                 # ~800 KB ⭐ FINAL MODEL
│   │
│   ├── preprocessing/                             # From Notebook 06
│   │   ├── scaler.pkl                            # StandardScaler object
│   │   └── label_encoder.pkl                     # Crop name encoder
│   │
│   ├── explainability/                            # From Notebook 13
│   │   └── shap_explainer.pkl                    # SHAP TreeExplainer
│   │
│   └── metadata/                                  # From Notebook 16
│       ├── model_metrics.json                    # All model accuracies
│       ├── feature_names.json                    # ['N','P','K',...]
│       ├── crop_labels.json                      # {0:'Rice', 1:'Wheat',...}
│       └── deployment_info.json                  # Version, date, params
│
├── 🌐 webapp/                                      # FLASK APPLICATION (.py ONLY)
│   │
│   ├── app.py                                     # ⭐ MAIN FLASK APP
│   │   # Routes:
│   │   # @app.route('/')                  → Home page (index.html)
│   │   # @app.route('/predict', POST)     → Crop prediction API
│   │   # @app.route('/explain', POST)     → SHAP explanation
│   │   # @app.route('/economic', POST)    → Economic analysis
│   │   # @app.route('/rotation', POST)    → Rotation planning (optional)
│   │   # @app.route('/compare', POST)     → Compare multiple crops
│   │   # @app.route('/about')             → About page
│   │   #
│   │   # Functions:
│   │   # - load_models()         → Load all .pkl files on startup
│   │   # - validate_request()    → Check input ranges
│   │   # - format_response()     → JSON formatting
│   │
│   ├── utils.py                                   # Utility functions
│   │   # Functions:
│   │   # - load_all_models() → Load model, scaler, encoder, explainer
│   │   # - validate_input(N, P, K, ...) → Range checks (N: 0-150, pH: 0-14)
│   │   # - prepare_features() → Scale input, reshape for model
│   │   # - format_output() → Convert predictions to JSON
│   │   # - error_handler() → Custom error messages
│   │
│   ├── prediction.py                              # Prediction logic
│   │   # Functions:
│   │   # - predict_crop(features) → Returns top 3 crops with probabilities
│   │   # - get_crop_probabilities() → All 22 crop probabilities
│   │   # - confidence_score() → Prediction confidence (0-100%)
│   │   # - categorize_recommendation():
│   │   #     * Recommended (prob > 70%)
│   │   #     * Slightly Recommended (40-70%)
│   │   #     * Not Recommended (< 40%)
│   │
│   ├── explainability.py                          # XAI logic
│   │   # Functions:
│   │   # - generate_shap_explanation(features) → SHAP values
│   │   # - create_shap_plots(shap_values):
│   │   #     * Waterfall plot (save as PNG)
│   │   #     * Force plot (save as HTML)
│   │   # - feature_contribution_text() → Human-readable explanation:
│   │   #     "Rice recommended because: High Rainfall (+0.35), 
│   │   #      Suitable Temperature (+0.28), Optimal pH (+0.22)"
│   │   # - get_top_features() → Top 5 contributing features
│   │
│   ├── economic.py                                # Economic analysis
│   │   # Functions:
│   │   # - calculate_roi(crop, season):
│   │   #     price = get_market_price(crop, season)
│   │   #     yield_estimate = 25  # quintals/acre (crop-specific)
│   │   #     revenue = price * yield_estimate
│   │   #     costs = seed + fertilizer + labor + misc
│   │   #     profit = revenue - costs
│   │   #     roi = (profit / costs) * 100
│   │   # - get_market_prices() → Load from crop_prices.csv
│   │   # - cost_benefit_analysis() → Revenue vs Cost breakdown
│   │   # - rank_by_profitability(crops) → Sort by ROI descending
│   │   # - risk_assessment() → Price volatility analysis
│   │
│   ├── rotation.py                                # Rotation planning (OPTIONAL)
│   │   # Functions:
│   │   # - get_rotation_suggestions(current_crop, season):
│   │   #     Load rotation_rules.csv
│   │   #     Filter compatible crops (compatibility_score > 0.7)
│   │   #     Return top 3 rotation options
│   │   # - plan_multiseason(crop):
│   │   #     Season 1: Recommended crop
│   │   #     Season 2: Compatible rotation (legume if cereal)
│   │   #     Season 3: Soil recovery crop
│   │   # - calculate_soil_impact():
│   │   #     Track N, P, K depletion/addition
│   │   #     Return soil health score (0-100)
│   │   # - sustainability_score() → Environmental rating
│   │
│   ├── templates/                                 # HTML TEMPLATES (Jinja2)
│   │   │
│   │   ├── base.html                              # Base template
│   │   │   # - Navigation bar (Home, Predict, About)
│   │   │   # - Footer
│   │   │   # - Bootstrap 5 CSS links
│   │   │   # - Chart.js, Plotly.js scripts
│   │   │   # - Block content for child templates
│   │   │
│   │   ├── index.html                             # Landing page
│   │   │   # - Hero section (project title, description)
│   │   │   # - Feature cards (ML, XAI, Economic, Rotation)
│   │   │   # - Statistics (22 crops, 98% accuracy, 3 innovations)
│   │   │   # - "Get Started" button → input_form.html
│   │   │
│   │   ├── input_form.html                        # Data input form
│   │   │   # Form fields:
│   │   │   # - Nitrogen (N): 0-150 kg/ha [slider + number input]
│   │   │   # - Phosphorus (P): 0-150 kg/ha
│   │   │   # - Potassium (K): 0-200 kg/ha
│   │   │   # - Temperature: 0-50°C
│   │   │   # - Humidity: 0-100%
│   │   │   # - pH: 3.5-9.0
│   │   │   # - Rainfall: 0-300 mm
│   │   │   # - Season: Kharif/Rabi/Zaid (dropdown)
│   │   │   # - Submit button → POST /predict
│   │   │   # Client-side validation (JavaScript)
│   │   │
│   │   ├── results.html                           # Prediction results
│   │   │   # Display:
│   │   │   # - Top 3 recommended crops (cards)
│   │   │   #   * Crop name + icon
│   │   │   #   * Confidence score (progress bar)
│   │   │   #   * Category badge (Recommended/Slightly/Not)
│   │   │   # - Probability chart (Chart.js bar chart)
│   │   │   # - Buttons:
│   │   │   #   * "View Explanation" → explanation.html
│   │   │   #   * "Economic Analysis" → economic_dashboard.html
│   │   │   #   * "Rotation Plan" → rotation_plan.html
│   │   │   #   * "Try Again" → input_form.html
│   │   │
│   │   ├── explanation.html                       # SHAP visualizations
│   │   │   # Display:
│   │   │   # - SHAP waterfall plot (embedded PNG)
│   │   │   # - Feature contribution text:
│   │   │   #   "Your soil is suitable for Rice because:
│   │   │   #    1. Rainfall (200mm) contributes +35%
│   │   │   #    2. Temperature (25°C) contributes +28%
│   │   │   #    3. pH (6.5) contributes +22%"
│   │   │   # - SHAP summary plot (all features)
│   │   │   # - Interactive force plot (Plotly)
│   │   │   # - Download explanation as PDF button
│   │   │
│   │   ├── economic_dashboard.html                # ROI analysis
│   │   │   # Display:
│   │   │   # - ROI comparison table (top 3 crops):
│   │   │   #   | Crop | Price | Revenue | Cost | Profit | ROI% |
│   │   │   # - Profitability ranking (sorted by ROI)
│   │   │   # - Cost breakdown chart (pie chart):
│   │   #   #   * Seed: 15%, Fertilizer: 35%, Labor: 40%, Misc: 10%
│   │   │   # - Revenue projection chart (bar chart)
│   │   │   # - Risk assessment badge (Low/Medium/High)
│   │   │   # - Market price trend (line chart, last 3 years)
│   │   │
│   │   ├── rotation_plan.html                     # Multi-season plan (OPTIONAL)
│   │   │   # Display:
│   │   │   # - 3-season timeline:
│   │   │   #   Season 1 (Current): Rice
│   │   │   #   Season 2 (Next): Wheat
│   │   │   #   Season 3 (Recovery): Mung Bean
│   │   │   # - Soil health tracker:
│   │   │   #   * Nitrogen: -20kg → -5kg → +15kg (recovery)
│   │   │   #   * Phosphorus: Stable
│   │   │   #   * Potassium: -10kg → +5kg
│   │   │   # - Sustainability score: 85/100 (Excellent)
│   │   │   # - Compatibility matrix (heatmap)
│   │   │   # - Benefits text for each rotation
│   │   │
│   │   └── comparison.html                        # Compare crops side-by-side
│   │       # Display:
│   │       # - Select 2-3 crops to compare
│   │       # - Comparison table:
│   │       #   | Feature | Rice | Wheat | Maize |
│   │       #   | Confidence | 92% | 78% | 65% |
│   │       #   | ROI | 45% | 38% | 52% |
│   │       #   | Risk | Low | Medium | Medium |
│   │       # - Radar chart (multi-dimensional comparison)
│   │       # - Recommendation: "Choose Maize for highest profit"
│   │
│   └── static/                                    # CSS, JS, Images
│       │
│       ├── css/
│       │   ├── bootstrap.min.css                  # Bootstrap 5.3
│       │   └── custom_styles.css                  # Custom CSS
│       │       # - Color scheme (green theme for agriculture)
│       │       # - Card hover effects
│       │       # - Responsive breakpoints
│       │       # - Chart container styling
│       │
│       ├── js/
│       │   ├── chart.min.js                       # Chart.js 4.4
│       │   ├── plotly.min.js                      # Plotly.js
│       │   │
│       │   ├── dashboard.js                       # Chart configurations
│       │   │   # Functions:
│       │   │   # - createProbabilityChart() → Bar chart for crop probabilities
│       │   │   # - createROIChart() → Comparison chart for economic analysis
│       │   │   # - createCostBreakdownChart() → Pie chart for costs
│       │   │   # - createRotationTimeline() → Timeline visualization
│       │   │
│       │   └── form_validation.js                 # Input validation
│       │       # Functions:
│       │       # - validateNumericInput() → Range checks
│       │       # - showErrorMessage() → Display validation errors
│       │       # - enableSubmitButton() → Enable after validation
│       │       # - syncSliderAndInput() → Link slider to number input
│       │
│       └── images/
│           ├── logo.png                           # Project logo
│           ├── hero_background.jpg                # Landing page image
│           │
│           ├── crop_icons/                        # Crop images (22 crops)
│           │   ├── rice.png
│           │   ├── wheat.png
│           │   ├── maize.png
│           │   └── ...
│           │
│           └── shap_plots/                        # Generated SHAP images
│               ├── waterfall_plot_1.png           # From explainability.py
│               ├── summary_plot.png
│               └── force_plot.html
│
├── 📄 docs/                                        # DOCUMENTATION
│   │
│   ├── project_report/
│   │   ├── 01_introduction.md
│   │   │   # - Problem statement
│   │   │   # - Objectives
│   │   │   # - Scope and limitations
│   │   │
│   │   ├── 02_literature_review.md
│   │   │   # - Review of 3 research papers
│   │   │   # - Identified gaps (XAI, Economic, Rotation)
│   │   │   # - Comparative analysis table
│   │   │
│   │   ├── 03_methodology.md
│   │   │   # - Dataset description
│   │   │   # - Data preprocessing steps
│   │   │   # - Feature engineering techniques
│   │   │   # - ML algorithms used
│   │   │   # - Evaluation metrics
│   │   │
│   │   ├── 04_results.md
│   │   │   # - Model performance comparison
│   │   │   # - Accuracy: 98.5% (Stacking Ensemble)
│   │   │   # - Confusion matrix analysis
│   │   │   # - SHAP interpretation insights
│   │   │   # - Economic analysis findings
│   │   │
│   │   ├── 05_conclusion.md
│   │   │   # - Summary of achievements
│   │   │   # - Innovations implemented
│   │   │   # - Limitations
│   │   │   # - Future work
│   │   │
│   │   └── final_report.pdf                      # Combined PDF (30-40 pages)
│   │
│   ├── presentation/
│   │   ├── project_presentation.pptx             # 15-20 slides
│   │   │   # Slide structure:
│   │   │   # 1. Title + Team
│   │   │   # 2. Problem Statement
│   │   │   # 3. Literature Review
│   │   │   # 4. Research Gaps
│   │   │   # 5. Proposed Solution
│   │   │   # 6. System Architecture
│   │   │   # 7-10. Methodology (Data, Features, Models)
│   │   │   # 11. Results (Model Comparison)
│   │   │   # 12. XAI Demo
│   │   │   # 13. Economic Module Demo
│   │   │   # 14. Web Application Screenshots
│   │   │   # 15. Innovations & Contributions
│   │   │   # 16. Conclusion & Future Work
│   │   │   # 17. Demo Video
│   │   │   # 18. Q&A
│   │   │
│   │   └── demo_video.mp4                        # 5-7 minute walkthrough
│   │
│   ├── user_manual/
│   │   └── user_guide.pdf                        # End-user documentation
│   │       # - How to use web application
│   │       # - Input parameter guidelines
│   │       # - Interpreting results
│   │       # - FAQs
│   │
│   └── api_documentation.md                      # Flask API reference
│       # Endpoint documentation:
│       # POST /predict
│       #   Request: {N, P, K, temp, humidity, ph, rainfall}
│       #   Response: {top_crops, probabilities, confidence}
│       # POST /explain
│       #   Request: {features}
│       #   Response: {shap_values, plot_url, text_explanation}
│       # POST /economic
│       #   Request: {crop, season}
│       #   Response: {roi, profit, cost_breakdown}
│
├── 🧪 tests/                                       # UNIT TESTS (Optional)
│   ├── test_prediction.py
│   │   # - test_predict_crop_valid_input()
│   │   # - test_predict_crop_invalid_input()
│   │   # - test_top_n_crops()
│   │
│   ├── test_economic.py
│   │   # - test_calculate_roi()
│   │   # - test_get_market_prices()
│   │
│   └── test_rotation.py
│       # - test_get_rotation_suggestions()
│       # - test_soil_impact_calculation()
│
├── ⚙️ config/                                      # CONFIGURATION
│   └── config.py
│       # Flask configuration:
│       # - SECRET_KEY
│       # - MODEL_PATH = '../models/ensemble/stacking_ensemble.pkl'
│       # - DATA_PATH = '../data/'
│       # - UPLOAD_FOLDER
│       # - MAX_CONTENT_LENGTH
│
├── 📦 requirements.txt                             # PYTHON DEPENDENCIES
│   # Core
│   flask==3.0.0
│   gunicorn==21.2.0
│   
│   # Jupyter
│   jupyter==1.0.0
│   notebook==7.0.6
│   ipywidgets==8.1.1
│   
│   # Data Science (Unit I)
│   pandas==2.1.0
│   numpy==1.24.3
│   openpyxl==3.1.2
│   
│   # Visualization (Unit II)
│   matplotlib==3.7.2
│   seaborn==0.12.2
│   plotly==5.16.1
│   
│   # Statistics (Unit II)
│   scipy==1.11.2
│   
│   # Machine Learning (Unit III, IV)
│   scikit-learn==1.3.0
│   xgboost==1.7.6
│   lightgbm==4.0.0
│   
│   # Explainability
│   shap==0.42.1
│   
│   # Utilities
│   joblib==1.3.2
│   python-dotenv==1.0.0
│
├── 📝 README.md                                    # PROJECT README
│   # Sections:
│   # - Project title & description
│   # - Features
│   # - Tech stack
│   # - Installation instructions
│   # - Usage guide
│   # - Dataset information
│   # - Model performance
│   # - Screenshots
│   # - Contributors
│   # - License
│
├── .gitignore                                     # Git ignore rules
│   # Ignore:
│   # - __pycache__/
│   # - *.pyc
│   # - venv/
│   # - .env
│   # - .ipynb_checkpoints/
│   # - *.pkl (models too large for Git)
│   # - data/processed/* (regenerated files)
│
└── LICENSE                                        # MIT License
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

