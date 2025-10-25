from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from disaster_model import DisasterRiskPredictor
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)

# Initialize the predictor
predictor = DisasterRiskPredictor()

# Load the trained model
try:
    predictor.load_model('disaster_model.pkl')
    model_loaded = True
except:
    model_loaded = False

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        # Prepare input data
        input_data = {
            'MonsoonIntensity': int(data['monsoon_intensity']),
            'TopographyDrainage': int(data['topography_drainage']),
            'RiverManagement': int(data['river_management']),
            'Deforestation': int(data['deforestation']),
            'Urbanization': int(data['urbanization']),
            'ClimateChange': int(data['climate_change']),
            'DamsQuality': int(data['dams_quality']),
            'Siltation': int(data['siltation']),
            'AgriculturalPractices': int(data['agricultural_practices']),
            'Encroachments': int(data['encroachments']),
            'IneffectiveDisasterPreparedness': int(data['disaster_preparedness']),
            'DrainageSystems': int(data['drainage_systems']),
            'CoastalVulnerability': int(data['coastal_vulnerability']),
            'Landslides': int(data['landslides']),
            'Watersheds': int(data['watersheds']),
            'DeterioratingInfrastructure': int(data['infrastructure']),
            'PopulationScore': int(data['population_score']),
            'WetlandLoss': int(data['wetland_loss']),
            'InadequatePlanning': int(data['planning']),
            'PoliticalFactors': int(data['political_factors'])
        }
        
        # Make prediction
        risk_probability, risk_class = predictor.predict_disaster_risk(input_data)
        
        # Determine risk level
        if risk_probability < 0.3:
            risk_level = "Low"
            risk_color = "#28a745"
        elif risk_probability < 0.7:
            risk_level = "Medium"
            risk_color = "#ffc107"
        else:
            risk_level = "High"
            risk_color = "#dc3545"
        
        return jsonify({
            'success': True,
            'risk_probability': float(risk_probability),
            'risk_class': int(risk_class),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': get_recommendations(risk_level, input_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        global predictor, model_loaded
        
        # Initialize new predictor
        predictor = DisasterRiskPredictor()
        
        # Load and train model
        X, y = predictor.load_and_preprocess_data('flood.csv')
        predictor.train_model(X, y)
        
        # Save model
        predictor.save_model()
        model_loaded = True
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_info')
def model_info():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        # Get feature coefficients
        coefficients = abs(predictor.model.coef_)
        feature_names = predictor.feature_names
        
        # Create feature coefficients chart
        fig = go.Figure(data=[
            go.Bar(
                x=coefficients,
                y=feature_names,
                orientation='h',
                marker_color='rgba(55, 128, 191, 0.7)',
                marker_line_color='rgba(55, 128, 191, 1.0)',
                marker_line_width=1
            )
        ])
        
        fig.update_layout(
            title='Feature Coefficients - Linear Regression Model',
            xaxis_title='Absolute Coefficient Value',
            yaxis_title='Features',
            height=600,
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'chart': graphJSON,
            'model_params': {
                'algorithm': 'Linear Regression',
                'n_features': len(feature_names),
                'intercept': float(predictor.model.intercept_)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_recommendations(risk_level, input_data):
    """Generate recommendations based on risk level and input parameters"""
    recommendations = []
    
    if risk_level == "High":
        recommendations.extend([
            "🚨 Immediate evacuation may be necessary",
            "📱 Monitor emergency alerts and weather updates",
            "🏠 Secure property and move valuables to higher ground",
            "🚗 Avoid travel in flood-prone areas",
            "💧 Store emergency water and food supplies"
        ])
        
        if input_data['MonsoonIntensity'] > 8:
            recommendations.append("⚠️ High monsoon intensity - expect heavy rainfall")
        
        if input_data['Urbanization'] > 7:
            recommendations.append("🏙️ High urbanization increases flood risk - avoid low areas")
            
    elif risk_level == "Medium":
        recommendations.extend([
            "⚠️ Stay alert and monitor weather conditions",
            "📦 Prepare emergency kit with essentials",
            "🏠 Check drainage systems around property",
            "📱 Keep emergency contacts readily available",
            "🚗 Plan alternative routes if needed"
        ])
        
    else:  # Low risk
        recommendations.extend([
            "✅ Current conditions indicate low flood risk",
            "📱 Continue monitoring weather updates",
            "🏠 Regular maintenance of drainage systems recommended",
            "📋 Review emergency preparedness plans",
            "🌱 Consider sustainable water management practices"
        ])
    
    # Add specific recommendations based on input parameters
    if input_data['Deforestation'] > 7:
        recommendations.append("🌲 High deforestation increases flood risk")
    
    if input_data['DeterioratingInfrastructure'] > 7:
        recommendations.append("🏗️ Poor infrastructure - consider evacuation routes")
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)