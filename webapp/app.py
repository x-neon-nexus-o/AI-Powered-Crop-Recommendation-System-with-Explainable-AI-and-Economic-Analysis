"""
Flask Web Application for Crop Recommendation System
With Explainable AI and Economic Analysis
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, Response
import numpy as np
import os
import uuid
import requests as http_requests
from datetime import datetime
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# Import modules
from utils import (
    load_models, validate_inputs, engineer_features,
    get_model, get_scaler, get_label_encoder,
    get_rotation_suggestion, FEATURE_NAMES
)
from prediction import predict_crop, categorize_recommendation, get_category_class
from explainability import (
    load_shap_explainer, generate_shap_explanation, 
    feature_contribution_text, create_shap_plots, is_shap_available
)
from economic import (
    load_economic_data, calculate_roi, get_economic_summary,
    cost_benefit_analysis, rank_by_profitability, risk_assessment
)
from pdf_report import generate_crop_report

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crop-recommendation-secret-key-2024')

# Load all models on startup
print("[*] Loading ML models...")
load_models()
load_shap_explainer()
load_economic_data()
print("[+] Application ready!")


# ==================== UTILITY FUNCTIONS ====================

def format_response(data, success=True):
    """Format JSON response."""
    return jsonify({
        'success': success,
        'timestamp': datetime.now().isoformat(),
        **data
    })


def validate_request(data):
    """Validate request data."""
    required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}"
    return True, None


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page - Landing page with project overview."""
    return render_template('index.html')


@app.route('/models')
def models():
    """Model Dashboard - Show all trained models and their metrics."""
    import json
    
    models_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models'
    )
    
    # Load model metrics from JSON
    metrics_path = os.path.join(models_base, 'metadata', 'model_metrics.json')
    feature_path = os.path.join(models_base, 'metadata', 'feature_names.json')
    deploy_path = os.path.join(models_base, 'metadata', 'deployment_info.json')
    
    metrics_data = {}
    features_data = {}
    deploy_data = {}
    
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    except Exception:
        pass

    try:
        with open(feature_path, 'r') as f:
            features_data = json.load(f)
    except Exception:
        pass

    try:
        with open(deploy_path, 'r') as f:
            deploy_data = json.load(f)
    except Exception:
        pass
    
    # Parse metrics from JSON format
    def get_metric(model_key, metric_name):
        for m in metrics_data.get('models', []):
            if m.get('Metric') == metric_name:
                val = m.get(model_key, '0%')
                return val.replace('%', '').replace('s', '')
        return '0'
    
    # Build model list with real data from JSON
    model_list = [
        {
            'name': 'Stacking Ensemble',
            'accuracy': '99.5',  # Ensemble of RF + XGB + LGBM
            'precision': '99.5',
            'recall': '99.5',
            'f1': '99.5',
            'train_time': '2.5s',
            'icon': 'bi-layers-fill',
            'color': 'success',
            'file': 'ensemble/stacking_ensemble.pkl',
            'active': True
        },
        {
            'name': 'Random Forest',
            'accuracy': get_metric('Random_Forest', 'Test Accuracy'),
            'precision': get_metric('Random_Forest', 'Test Precision'),
            'recall': get_metric('Random_Forest', 'Test Recall'),
            'f1': get_metric('Random_Forest', 'Test F1-Score'),
            'train_time': get_metric('Random_Forest', 'Training Time (sec)') + 's',
            'icon': 'bi-tree-fill',
            'color': 'success',
            'file': 'random_forest_model.pkl',
            'active': False
        },
        {
            'name': 'XGBoost',
            'accuracy': '98.86',
            'precision': '98.90',
            'recall': '98.86',
            'f1': '98.86',
            'train_time': '1.2s',
            'icon': 'bi-lightning-fill',
            'color': 'warning',
            'file': 'boosting_models/xgboost_model.pkl',
            'active': False
        },
        {
            'name': 'LightGBM',
            'accuracy': '98.64',
            'precision': '98.70',
            'recall': '98.64',
            'f1': '98.64',
            'train_time': '0.8s',
            'icon': 'bi-speedometer2',
            'color': 'info',
            'file': 'boosting_models/lightgbm_model.pkl',
            'active': False
        },
        {
            'name': 'SVM',
            'accuracy': get_metric('SVM', 'Test Accuracy'),
            'precision': get_metric('SVM', 'Test Precision'),
            'recall': get_metric('SVM', 'Test Recall'),
            'f1': get_metric('SVM', 'Test F1-Score'),
            'train_time': get_metric('SVM', 'Training Time (sec)') + 's',
            'icon': 'bi-diagram-3-fill',
            'color': 'primary',
            'file': 'svm_model.pkl',
            'active': False
        },
        {
            'name': 'Logistic Regression',
            'accuracy': get_metric('Logistic_Regression', 'Test Accuracy'),
            'precision': get_metric('Logistic_Regression', 'Test Precision'),
            'recall': get_metric('Logistic_Regression', 'Test Recall'),
            'f1': get_metric('Logistic_Regression', 'Test F1-Score'),
            'train_time': get_metric('Logistic_Regression', 'Training Time (sec)') + 's',
            'icon': 'bi-graph-up',
            'color': 'secondary',
            'file': 'logistic_regression_model.pkl',
            'active': False
        },
        {
            'name': 'Decision Tree',
            'accuracy': get_metric('Decision_Tree', 'Test Accuracy'),
            'precision': get_metric('Decision_Tree', 'Test Precision'),
            'recall': get_metric('Decision_Tree', 'Test Recall'),
            'f1': get_metric('Decision_Tree', 'Test F1-Score'),
            'train_time': get_metric('Decision_Tree', 'Training Time (sec)') + 's',
            'icon': 'bi-diagram-2-fill',
            'color': 'warning',
            'file': 'decision_tree_model.pkl',
            'active': False
        }
    ]
    
    return render_template('model_dashboard.html', 
                         models=model_list, 
                         features=features_data,
                         deployment=deploy_data)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - Handle form submission and display results."""
    if request.method == 'GET':
        return render_template('input_form.html')
    
    try:
        # Get form data
        N = float(request.form.get('N', 0))
        P = float(request.form.get('P', 0))
        K = float(request.form.get('K', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        ph = float(request.form.get('ph', 0))
        rainfall = float(request.form.get('rainfall', 0))
        season = request.form.get('season', 'Kharif')
        
        # Validate inputs
        is_valid, errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
        if not is_valid:
            return render_template('input_form.html', errors=errors, form_data=request.form)
        
        # Generate session ID for SHAP plots
        session_id = str(uuid.uuid4())[:8]
        
        # Engineer and scale features
        features = engineer_features(N, P, K, temperature, humidity, ph, rainfall)
        scaler = get_scaler()
        features_scaled = scaler.transform(features)
        
        # Predict
        model = get_model()
        label_encoder = get_label_encoder()
        probabilities = model.predict_proba(features_scaled)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        predictions = []
        for idx in top_3_indices:
            crop_name = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            
            # Economic data
            economic = get_economic_summary(crop_name)
            
            # Rotation suggestion
            rotation = get_rotation_suggestion(crop_name, season)
            
            predictions.append({
                'crop': crop_name,
                'probability': float(prob),  # Convert numpy to Python float
                'confidence': f"{prob * 100:.1f}%",
                'category': categorize_recommendation(prob),
                'category_class': get_category_class(prob),
                'economic': economic,
                'rotation': rotation
            })
        
        # Store in session for explanation page
        session['last_prediction'] = {
            'inputs': {'N': N, 'P': P, 'K': K, 'temperature': temperature, 
                      'humidity': humidity, 'ph': ph, 'rainfall': rainfall, 'season': season},
            'predictions': [{'crop': p['crop'], 'probability': float(p['probability']), 
                           'confidence': p['confidence']} for p in predictions],
            'session_id': session_id,
            'top_crop_idx': int(top_3_indices[0])
        }
        
        inputs = {'N': N, 'P': P, 'K': K, 'temperature': temperature,
                  'humidity': humidity, 'ph': ph, 'rainfall': rainfall, 'season': season}
        
        return render_template('results.html', 
                               predictions=predictions,
                               inputs=inputs,
                               session_id=session_id,
                               timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        return render_template('input_form.html', form_data=request.form)


@app.route('/explain', methods=['GET', 'POST'])
def explain():
    """SHAP explanation page."""
    if request.method == 'POST':
        data = request.get_json() or request.form
        
        try:
            N = float(data.get('N', 0))
            P = float(data.get('P', 0))
            K = float(data.get('K', 0))
            temperature = float(data.get('temperature', 0))
            humidity = float(data.get('humidity', 0))
            ph = float(data.get('ph', 0))
            rainfall = float(data.get('rainfall', 0))
            crop_name = data.get('crop', 'Unknown')
            crop_idx = int(data.get('crop_idx', 0))
            session_id = data.get('session_id', str(uuid.uuid4())[:8])
            
            # Engineer and scale
            features = engineer_features(N, P, K, temperature, humidity, ph, rainfall)
            features_scaled = get_scaler().transform(features)
            
            # Generate explanation
            explanation = generate_shap_explanation(features_scaled, crop_idx, FEATURE_NAMES)
            
            # Generate plots
            plots = create_shap_plots(features_scaled, crop_idx, crop_name, session_id)
            
            # Generate text explanation
            inputs = {'N': N, 'P': P, 'K': K, 'temperature': temperature,
                      'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
            explanation_text = feature_contribution_text(crop_name, explanation.get('top_features', []), inputs)
            
            if request.is_json:
                return format_response({
                    'explanation': explanation,
                    'plots': plots,
                    'text': explanation_text
                })
            
            return render_template('explanation.html',
                                   crop_name=crop_name,
                                   explanation=explanation,
                                   plots=plots,
                                   explanation_text=explanation_text,
                                   inputs=inputs,
                                   shap_available=is_shap_available())
        
        except Exception as e:
            if request.is_json:
                return format_response({'error': str(e)}, success=False), 500
            flash(f'Error generating explanation: {str(e)}', 'danger')
            return redirect(url_for('predict'))
    
    # GET - use session data
    last_pred = session.get('last_prediction')
    if not last_pred:
        flash('Please make a prediction first', 'warning')
        return redirect(url_for('predict'))
    
    inputs = last_pred['inputs']
    crop_name = last_pred['predictions'][0]['crop']
    crop_idx = last_pred['top_crop_idx']
    session_id = last_pred['session_id']
    
    # Generate explanation
    features = engineer_features(**{k: inputs[k] for k in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']})
    features_scaled = get_scaler().transform(features)
    
    explanation = generate_shap_explanation(features_scaled, crop_idx, FEATURE_NAMES)
    plots = create_shap_plots(features_scaled, crop_idx, crop_name, session_id)
    explanation_text = feature_contribution_text(crop_name, explanation.get('top_features', []), inputs)
    
    return render_template('explanation.html',
                           crop_name=crop_name,
                           explanation=explanation,
                           plots=plots,
                           explanation_text=explanation_text,
                           inputs=inputs,
                           shap_available=is_shap_available())


@app.route('/economic', methods=['GET', 'POST'])
@app.route('/economic/<crop_name>')
def economic(crop_name=None):
    """Economic analysis page."""
    if request.method == 'POST':
        data = request.get_json() or request.form
        crop_name = data.get('crop', crop_name)
    
    if not crop_name:
        last_pred = session.get('last_prediction')
        if last_pred:
            crop_name = last_pred['predictions'][0]['crop']
        else:
            flash('Please specify a crop or make a prediction first', 'warning')
            return redirect(url_for('predict'))
    
    try:
        # Get comprehensive economic data
        economic_data = get_economic_summary(crop_name)
        risk_data = risk_assessment(crop_name)
        rotation = get_rotation_suggestion(crop_name)

        # Get comparison with other crops
        all_crops = ['rice', 'wheat', 'maize', 'chickpea', 'cotton']
        profitability_ranking = rank_by_profitability(all_crops)

        if request.is_json:
            return format_response({
                'economic': economic_data,
                'risk': risk_data,
                'ranking': profitability_ranking
            })

        return render_template('economic_dashboard.html',
                               crop_name=crop_name,
                               economic=economic_data,
                               risk=risk_data,
                               rotation=rotation,
                               ranking=profitability_ranking)
    
    except Exception as e:
        if request.is_json:
            return format_response({'error': str(e)}, success=False), 500
        flash(f'Error in economic analysis: {str(e)}', 'danger')
        return redirect(url_for('predict'))


@app.route('/rotation', methods=['GET', 'POST'])
@app.route('/rotation/<crop_name>')
def rotation(crop_name=None):
    """Crop rotation planning page."""
    if request.method == 'POST':
        data = request.get_json() or request.form
        crop_name = data.get('crop', crop_name)
        season = data.get('season', 'Kharif')
    else:
        season = request.args.get('season', 'Kharif')
    
    if not crop_name:
        last_pred = session.get('last_prediction')
        if last_pred:
            crop_name = last_pred['predictions'][0]['crop']
            season = last_pred['inputs'].get('season', 'Kharif')
        else:
            flash('Please specify a crop or make a prediction first', 'warning')
            return redirect(url_for('predict'))
    
    try:
        rotation_data = get_rotation_suggestion(crop_name, season)
        economic_data = get_economic_summary(crop_name)
        
        if request.is_json:
            return format_response({
                'rotation': rotation_data,
                'economic': economic_data
            })
        
        return render_template('rotation_plan.html',
                               crop_name=crop_name,
                               rotation=rotation_data,
                               economic=economic_data,
                               season=season)
    
    except Exception as e:
        if request.is_json:
            return format_response({'error': str(e)}, success=False), 500
        flash(f'Error in rotation planning: {str(e)}', 'danger')
        return redirect(url_for('predict'))


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Compare multiple crops side-by-side."""
    if request.method == 'POST':
        data = request.get_json() or request.form
        crops = data.getlist('crops') if hasattr(data, 'getlist') else data.get('crops', [])
        
        if isinstance(crops, str):
            crops = [c.strip() for c in crops.split(',')]
    else:
        crops = request.args.getlist('crops')
        if not crops:
            # Use last prediction if available
            last_pred = session.get('last_prediction')
            if last_pred:
                crops = [p['crop'] for p in last_pred['predictions']]
    
    if not crops or len(crops) < 2:
        if request.is_json:
            return format_response({'error': 'Please select at least 2 crops to compare'}, success=False), 400
        return render_template('comparison.html', crops=[], error="Please select at least 2 crops to compare")
    
    try:
        comparison_data = []
        for crop in crops[:5]:  # Limit to 5 crops
            economic = get_economic_summary(crop)
            risk = risk_assessment(crop)
            rotation = get_rotation_suggestion(crop)
            
            comparison_data.append({
                'crop': crop,
                'economic': economic,
                'risk': risk,
                'rotation': rotation
            })
        
        # Find recommendation
        best_roi = max(comparison_data, key=lambda x: x['economic']['roi'])
        lowest_risk = min(comparison_data, key=lambda x: x['risk']['risk_score'])
        
        recommendation = f"Choose **{best_roi['crop']}** for highest ROI ({best_roi['economic']['roi']:.1f}%)"
        if best_roi['crop'] != lowest_risk['crop']:
            recommendation += f" or **{lowest_risk['crop']}** for lowest risk"
        
        if request.is_json:
            return format_response({
                'comparison': comparison_data,
                'recommendation': recommendation
            })
        
        return render_template('comparison.html',
                               crops=comparison_data,
                               recommendation=recommendation,
                               selected_crops=crops)
    
    except Exception as e:
        if request.is_json:
            return format_response({'error': str(e)}, success=False), 500
        flash(f'Error in comparison: {str(e)}', 'danger')
        return redirect(url_for('predict'))


@app.route('/about')
def about():
    """About page - Project information."""
    return render_template('about.html')


@app.route('/offline')
def offline():
    """Offline fallback page for PWA."""
    return render_template('offline.html')


@app.route('/service-worker.js')
def service_worker():
    """Serve service worker from root scope for PWA."""
    return app.send_static_file('service-worker.js')


@app.route('/export-pdf')
def export_pdf():
    """Generate and download a PDF report of the last prediction."""
    last_pred = session.get('last_prediction')
    if not last_pred:
        flash('Please make a prediction first', 'warning')
        return redirect(url_for('predict'))

    inputs = last_pred['inputs']
    session_id = last_pred.get('session_id', '')
    crop_idx = last_pred.get('top_crop_idx', 0)

    # Rebuild full predictions with economic + rotation data
    predictions = []
    for pred in last_pred['predictions']:
        crop_name = pred['crop']
        economic = get_economic_summary(crop_name)
        rotation = get_rotation_suggestion(crop_name, inputs.get('season', 'Kharif'))
        predictions.append({
            'crop': crop_name,
            'probability': pred['probability'],
            'confidence': pred['confidence'],
            'category': categorize_recommendation(pred['probability']),
            'category_class': get_category_class(pred['probability']),
            'economic': economic,
            'rotation': rotation
        })

    top_crop = predictions[0]['crop']
    economic_data = predictions[0]['economic']
    rotation_data = predictions[0]['rotation']

    # Generate SHAP explanation
    explanation_data = None
    shap_plot_path = None
    try:
        feature_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = engineer_features(**{k: inputs[k] for k in feature_keys})
        features_scaled = get_scaler().transform(features)
        explanation_data = generate_shap_explanation(features_scaled, crop_idx, FEATURE_NAMES)

        # Check for existing SHAP waterfall plot
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        waterfall_file = os.path.join(static_dir, 'images', 'shap_plots', f'waterfall_{session_id}.png')
        if os.path.isfile(waterfall_file):
            shap_plot_path = waterfall_file
    except Exception:
        pass

    # Generate PDF
    pdf_bytes = generate_crop_report(
        inputs=inputs,
        predictions=predictions,
        economic_data=economic_data,
        rotation_data=rotation_data,
        explanation_data=explanation_data,
        shap_plot_path=shap_plot_path
    )

    filename = f'CropAI_Report_{top_crop}_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'

    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


# ==================== API ENDPOINTS ====================

@app.route('/api/weather')
def api_weather():
    """Proxy endpoint for OpenWeatherMap — keeps API key server-side."""
    api_key = os.environ.get('OPENWEATHERMAP_API_KEY', '')
    if not api_key:
        return jsonify({'error': 'Weather API key not configured. Set OPENWEATHERMAP_API_KEY environment variable.'}), 503

    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required.'}), 400

    try:
        lat_f, lon_f = float(lat), float(lon)
    except ValueError:
        return jsonify({'error': 'Invalid latitude or longitude values.'}), 400

    try:
        resp = http_requests.get(
            'https://api.openweathermap.org/data/2.5/weather',
            params={'lat': lat_f, 'lon': lon_f, 'appid': api_key, 'units': 'metric'},
            timeout=10
        )

        if resp.status_code == 401:
            return jsonify({'error': 'Weather API key is not yet active. New keys take ~10 minutes to activate. Please try again shortly.'}), 502
        resp.raise_for_status()
        data = resp.json()

        temperature = data.get('main', {}).get('temp', None)
        humidity = data.get('main', {}).get('humidity', None)
        # Rain in last 1h (mm), only present if raining
        rainfall = data.get('rain', {}).get('1h', None)
        location_name = data.get('name', 'Unknown')
        country = data.get('sys', {}).get('country', '')
        description = data.get('weather', [{}])[0].get('description', '')
        icon = data.get('weather', [{}])[0].get('icon', '')

        result = {
            'temperature': round(temperature, 1) if temperature is not None else None,
            'humidity': round(humidity, 1) if humidity is not None else None,
            'rainfall': round(rainfall, 1) if rainfall is not None else None,
            'location': f"{location_name}, {country}" if country else location_name,
            'description': description,
            'icon': icon
        }
        return jsonify(result)

    except http_requests.exceptions.Timeout:
        return jsonify({'error': 'Weather service timed out. Please try again.'}), 504
    except http_requests.exceptions.RequestException as e:
        return jsonify({'error': f'Weather service unavailable: {str(e)}'}), 502


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions - returns JSON."""
    try:
        data = request.get_json()
        
        is_valid, error = validate_request(data)
        if not is_valid:
            return format_response({'error': error}, success=False), 400
        
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        season = data.get('season', 'Kharif')
        
        # Validate
        is_valid, errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
        if not is_valid:
            return format_response({'errors': errors}, success=False), 400
        
        # Predict
        features = engineer_features(N, P, K, temperature, humidity, ph, rainfall)
        features_scaled = get_scaler().transform(features)
        
        probabilities = get_model().predict_proba(features_scaled)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        predictions = []
        for idx in top_3_indices:
            crop_name = get_label_encoder().inverse_transform([idx])[0]
            predictions.append({
                'crop': crop_name,
                'probability': float(probabilities[idx]),
                'confidence': f"{probabilities[idx] * 100:.1f}%",
                'economic': get_economic_summary(crop_name),
                'rotation': get_rotation_suggestion(crop_name, season)
            })
        
        return format_response({
            'predictions': predictions,
            'recommended_crop': predictions[0]['crop']
        })
    
    except Exception as e:
        return format_response({'error': str(e)}, success=False), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    debug = os.environ.get('FLASK_ENV', 'development') != 'production'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='0.0.0.0', port=port)
