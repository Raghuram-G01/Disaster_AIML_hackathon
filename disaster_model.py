import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class DisasterRiskPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the flood dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('FloodProbability', axis=1)
        y = df['FloodProbability']  # Keep as continuous values for regression
        
        self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"Dataset shape: {X_scaled.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Feature importance
        self.plot_feature_importance()
        
        return X_test, y_test, y_pred
    
    def plot_feature_importance(self):
        """Plot feature coefficients for Linear Regression"""
        coefficients = abs(self.model.coef_)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), x='coefficient', y='feature')
        plt.title('Top 10 Feature Coefficients - Linear Regression')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def predict_disaster_risk(self, input_data):
        """Predict disaster risk for new data"""
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Scale the data
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        risk_probability = self.model.predict(input_scaled)[0]
        
        # Ensure probability is between 0 and 1
        risk_probability = max(0, min(1, risk_probability))
        
        # Convert to binary classification
        risk_class = 1 if risk_probability > 0.5 else 0
        
        return risk_probability, risk_class
    
    def save_model(self, model_path='disaster_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='disaster_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {model_path}")

def main():
    # Initialize predictor
    predictor = DisasterRiskPredictor()
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data('flood.csv')
    
    # Train model
    X_test, y_test, y_pred = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    # Example prediction
    sample_data = {
        'MonsoonIntensity': 7,
        'TopographyDrainage': 5,
        'RiverManagement': 6,
        'Deforestation': 4,
        'Urbanization': 6,
        'ClimateChange': 5,
        'DamsQuality': 5,
        'Siltation': 4,
        'AgriculturalPractices': 5,
        'Encroachments': 5,
        'IneffectiveDisasterPreparedness': 6,
        'DrainageSystems': 5,
        'CoastalVulnerability': 4,
        'Landslides': 5,
        'Watersheds': 5,
        'DeterioratingInfrastructure': 5,
        'PopulationScore': 6,
        'WetlandLoss': 5,
        'InadequatePlanning': 5,
        'PoliticalFactors': 5
    }
    
    risk_prob, risk_class = predictor.predict_disaster_risk(sample_data)
    print(f"\nSample Prediction:")
    print(f"Risk Probability: {risk_prob:.4f}")
    print(f"Risk Class: {'High Risk' if risk_class == 1 else 'Low Risk'}")

if __name__ == "__main__":
    main()