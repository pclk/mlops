from __future__ import annotations

import bentoml
import numpy as np
import pandas as pd
from pycaret.regression import load_model
from pydantic import BaseModel
from typing import Dict, List
from brand_model_frequency import brand_model_frequency

# Define the Pydantic model for car input features
class CarInput(BaseModel):
    brand_model: str
    location: str
    year: int
    kilometers_driven: float
    fuel_type: str
    transmission: str
    owner_type: str
    mileage: float
    power: float
    seats: int

# Configure BentoML image with required packages
bento_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("pycaret", "numpy", "pandas", "scikit-learn", "catboost")

@bentoml.service(
    image=bento_image,
    resources={"cpu": "2"},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
        }
    }
)

class CarPricePredictor:
    def __init__(self) -> None:
        self.model_pipeline = load_model("car_price_pipeline")
        print("✅ PyCaret pipeline loaded successfully!")

    # Helper method for preprocessing input
    def preprocess_input(self, data: CarInput) -> pd.DataFrame:
        data_dict = data.model_dump()
        print("🚗 Data Received:", data_dict)

        # Log1p transformation for skewed features
        data_dict["kilometers_driven"] = np.log1p(data_dict["kilometers_driven"])
        data_dict["mileage"] = np.log1p(data_dict["mileage"])
        data_dict["power"] = np.log1p(data_dict["power"])
        print("Data Logged: ", data_dict)

        # Encode Brand_Model using frequency dictionary (default to 1)
        brand_model_freq = brand_model_frequency.get(data_dict["brand_model"], 1)
        print("Brand Model Frequency: ", brand_model_freq)

        # Label encode Owner_Type
        owner_type_mapping = {"First": 0, "Second": 1, "Third & Above": 2}
        owner_type_encoded = owner_type_mapping.get(data_dict["owner_type"], 2)

        input_df = pd.DataFrame([{
            "Brand_Model_Encoded": brand_model_freq,
            "Location": data_dict["location"],
            "Year": data_dict["year"],
            "Kilometers_Driven": data_dict["kilometers_driven"],
            "Fuel_Type": data_dict["fuel_type"],
            "Transmission": data_dict["transmission"],
            "Owner_Type": owner_type_encoded,
            "Mileage": data_dict["mileage"],
            "Power": data_dict["power"],
            "Seats": data_dict["seats"]
        }])

        print("📊 Preprocessed DataFrame:\n", input_df)
        return input_df

    @bentoml.api(batchable=True)
    def predict(self, input: List[CarInput]) -> List[Dict[str, float]]:
        predictions = []
        for data in input:
            try:
                # Preprocess input and make prediction
                input_df = self.preprocess_input(data)
                prediction = self.model_pipeline.predict(input_df)
                predicted_price = round(float(np.expm1(prediction[0])), 2)
                predictions.append({"predicted_price": predicted_price})
            except Exception as e:
                predictions.append({"error": str(e)})

        print("🔮 Predictions:", predictions)
        return predictions
