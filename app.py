from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import os
import logging
import random
import gdown
import joblib
# For generating variable dummy predictions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Initialize FastAPI app
app = FastAPI(title="Air Quality Prediction API",
              description="API for predicting air quality based on environmental factors",
              version="1.0.0")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directory for models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)  # Create models directory if it doesn't exist


# File ID from your Google Drive link
file_id = "1UiL9PPE8tJ_JnUQtzWd1MQ66Sn8dVGIG"
output = "model.joblib"

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load the model
model = joblib.load(output)
print("Model loaded successfully!")

# Try to load the model from different file formats
try:
    logger.info(f"Attempting to load model from: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model.joblib: {e}")
    try:
        logger.info(f"Attempting to load model from: {model_pkl_path}")
        model = joblib.load(model_pkl_path)
        logger.info(f"Model loaded successfully from {model_pkl_path}")
    except Exception as e_pkl:
        logger.error(f"Error loading model.pkl: {e_pkl}")
        logger.warning("No model loaded, will use dummy predictions")

# Define input data model
class AirQualityInput(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    temperature: float
    humidity: float
    wind_speed: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "pm25": 20.5,
                "pm10": 45.3,
                "no2": 25.1,
                "so2": 10.2,
                "co": 0.8,
                "o3": 35.7,
                "temperature": 24.5,
                "humidity": 60.0,
                "wind_speed": 5.2
            }
        }

# Define AQI categories
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return {"category": "Good", "color": "green", "description": "Air quality is satisfactory, and air pollution poses little or no risk."}
    elif aqi_value <= 100:
        return {"category": "Moderate", "color": "yellow", "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."}
    elif aqi_value <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "color": "orange", "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."}
    elif aqi_value <= 200:
        return {"category": "Unhealthy", "color": "red", "description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."}
    elif aqi_value <= 300:
        return {"category": "Very Unhealthy", "color": "purple", "description": "Health alert: The risk of health effects is increased for everyone."}
    else:
        return {"category": "Hazardous", "color": "maroon", "description": "Health warning of emergency conditions: everyone is more likely to be affected."}

# Generate a more realistic dummy prediction based on input data
def generate_dummy_prediction(data):
    # Create a weighted score based on input parameters
    # Higher pollutant values lead to higher AQI, temperature and wind affect it
    weighted_score = (
        data.pm25 * 0.3 +
        data.pm10 * 0.2 +
        data.no2 * 0.15 +
        data.so2 * 0.1 +
        data.co * 15 +  # CO is usually in ppm, so multiply for comparable scale
        data.o3 * 0.15
    )
    
    # Add some random variation (±10%)
    variation = random.uniform(0.9, 1.1)
    
    # Adjust based on weather factors - lower temp and higher wind speed improve air quality
    weather_factor = 1.0
    if data.temperature < 20:
        weather_factor *= 0.9
    elif data.temperature > 30:
        weather_factor *= 1.1
        
    if data.wind_speed > 8:
        weather_factor *= 0.85
    elif data.wind_speed < 3:
        weather_factor *= 1.15
        
    # Calculate final AQI - scale to reasonable AQI range
    aqi_value = min(300, max(30, weighted_score * variation * weather_factor))
    
    return float(aqi_value)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Air Quality Prediction API"}

@app.post("/predict/")
def predict_air_quality(data: AirQualityInput):
    # Log the received data for debugging
    logger.info(f"Received prediction request with data: {data.dict()}")
    
    if model is None:
        # If model is not available, return a dynamic dummy prediction based on input
        logger.warning("Model not loaded, returning dynamic dummy prediction")
        aqi_value = generate_dummy_prediction(data)
        category_info = get_aqi_category(aqi_value)
        
        logger.info(f"Generated dummy prediction: {aqi_value:.2f}")
        
        return {
            "predicted_aqi": aqi_value,
            "category": category_info["category"],
            "color": category_info["color"],
            "description": category_info["description"],
            "note": "This is a simulated prediction as the model is not loaded correctly."
        }
    
    try:
        # Convert input data to DataFrame (match the format expected by your model)
        input_data = pd.DataFrame({
            'PM2.5': [data.pm25],
            'PM10': [data.pm10],
            'NO2': [data.no2],
            'SO2': [data.so2],
            'CO': [data.co],
            'O3': [data.o3],
            'Temperature': [data.temperature],
            'Humidity': [data.humidity],
            'Wind_speed': [data.wind_speed]
        })
        
        logger.info(f"Prepared input data for model: {input_data.to_dict()}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        aqi_value = float(prediction)  # Convert numpy type to native Python type
        
        logger.info(f"Model predicted AQI: {aqi_value}")
        
        # Get AQI category
        category_info = get_aqi_category(aqi_value)
        
        return {
            "predicted_aqi": aqi_value,
            "category": category_info["category"],
            "color": category_info["color"],
            "description": category_info["description"]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fall back to dummy prediction on error
        aqi_value = generate_dummy_prediction(data)
        category_info = get_aqi_category(aqi_value)
        
        logger.info(f"Falling back to dummy prediction: {aqi_value:.2f}")
        
        return {
            "predicted_aqi": aqi_value,
            "category": category_info["category"],
            "color": category_info["color"],
            "description": category_info["description"],
            "note": f"Error occurred during prediction: {str(e)}. Using fallback prediction."
        }

# Sample historical data endpoint
@app.get("/historical-data/")
def get_historical_data():
    # In a real application, this would fetch data from a database
    # For demo purposes, we'll return some sample data
    return {
        "dates": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", 
                  "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10"],
        "aqi_values": [45, 62, 78, 56, 89, 102, 76, 45, 67, 55],
        "pm25_values": [15, 22, 28, 18, 35, 42, 30, 15, 25, 20],
        "pm10_values": [30, 45, 60, 40, 70, 85, 55, 30, 50, 40],
    }

# Model performance metrics endpoint
@app.get("/model-performance/")
def get_model_performance():
    # In a real application, these would be actual metrics from your model evaluation
    return {
        "mse": 15.23,
        "rmse": 3.90,
        "mae": 3.12,
        "r2": 0.85,
        "feature_importance": {
            "PM2.5": 0.35,
            "PM10": 0.20,
            "NO2": 0.15,
            "SO2": 0.10,
            "CO": 0.08,
            "O3": 0.07,
            "Temperature": 0.02,
            "Humidity": 0.02,
            "Wind_speed": 0.01
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
