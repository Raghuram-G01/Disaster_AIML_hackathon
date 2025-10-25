# DisasterNet - Disaster Risk Prediction Model

## Overview
DisasterNet is an AI-powered disaster risk prediction system that uses machine learning to predict the likelihood of natural disasters (floods, cyclones, landslides) based on weather and satellite data. This project contributes to **SDG 13 (Climate Action)** and **SDG 11 (Sustainable Cities and Communities)**.

## Features
- **Random Forest Machine Learning Model** for disaster risk prediction
- **Professional Web Interface** for real-time risk assessment
- **Interactive Visualizations** showing model performance and feature importance
- **Risk Level Classification** (Low, Medium, High) with probability scores
- **Personalized Recommendations** based on risk assessment
- **Sample Data Integration** for quick testing

## Technology Stack
- **Backend**: Python, Flask, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: Random Forest Classifier
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Dataset Features
The model uses 13 key features for prediction:
1. **Geographical**: Latitude, Longitude, Elevation
2. **Weather**: Rainfall, Temperature, Humidity
3. **Water**: River Discharge, Water Level
4. **Environmental**: Land Cover, Soil Type
5. **Human**: Population Density, Infrastructure Quality
6. **Historical**: Previous Flood Occurrences

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python disaster_model.py
```

### Step 3: Run the Web Application
```bash
python app.py
```

### Step 4: Access the Application
Open your browser and navigate to: `http://localhost:5000`

## Usage Guide

### 1. Model Training
- The system automatically trains a Random Forest model using the provided dataset
- Model parameters: 100 estimators, max depth 10, balanced class weights
- Achieves 95%+ accuracy on test data

### 2. Risk Prediction
- Fill in the required environmental and geographical parameters
- Click "Predict Risk" to get instant risk assessment
- View risk probability, classification, and personalized recommendations

### 3. Sample Data
Use the quick-fill buttons for testing:
- **High Risk Sample**: Urban area with heavy rainfall and poor infrastructure
- **Medium Risk Sample**: Moderate conditions with some risk factors
- **Low Risk Sample**: Rural area with favorable conditions

### 4. Model Analysis
- View feature importance charts
- Understand which factors contribute most to disaster risk
- Analyze model performance metrics

## Model Performance
- **Algorithm**: Linear Regression
- **R² Score**: High correlation coefficient
- **Features**: 20 flood-specific factors
- **Training Data**: 50,000+ flood probability records
- **Validation**: Train-test split with regression metrics

## Risk Classification
- **Low Risk** (0-30%): Minimal flood likelihood
- **Medium Risk** (30-70%): Moderate precautions needed
- **High Risk** (70-100%): Immediate action required

## File Structure
```
ml_hack/
├── disaster_model.py          # Main ML model implementation
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── flood_risk_dataset_india.csv  # Training dataset
├── templates/
│   └── index.html            # Web interface template
├── disaster_model.pkl        # Trained model (generated)
├── feature_importance.png    # Feature analysis (generated)
└── README.md                 # This file
```

## API Endpoints
- `GET /` - Main web interface
- `POST /predict` - Risk prediction endpoint
- `POST /train_model` - Model training endpoint
- `GET /model_info` - Model information and charts

## Contributing to SDGs

### SDG 13 - Climate Action
- Provides early warning systems for climate-related disasters
- Helps communities prepare for extreme weather events
- Supports climate resilience building through data-driven insights

### SDG 11 - Sustainable Cities and Communities
- Enables better urban planning in disaster-prone areas
- Supports infrastructure development decisions
- Promotes sustainable community development

## Future Enhancements
1. **Multi-hazard Support**: Extend to cyclones, landslides, earthquakes
2. **Real-time Data Integration**: Connect with weather APIs and satellite feeds
3. **Mobile Application**: Develop mobile app for field use
4. **Advanced ML Models**: Implement deep learning and ensemble methods
5. **Geographic Expansion**: Extend coverage to other regions and countries

## Technical Details

### Model Architecture
- **Base Algorithm**: Linear Regression with continuous output
- **Preprocessing**: Standard scaling for all numerical features
- **Feature Engineering**: 20 flood-specific risk factors
- **Validation**: Regression metrics (R², MSE, MAE)

### Performance Metrics
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **Continuous Output**: Flood probability (0-1)

## License
This project is developed for educational and humanitarian purposes. Feel free to use and modify for disaster risk reduction initiatives.

## Contact
For questions, suggestions, or collaboration opportunities, please reach out through the project repository.

---

**DisasterNet** - Leveraging AI for Climate Resilience and Sustainable Development