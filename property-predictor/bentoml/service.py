# service.py
import bentoml


bento_image = bentoml.images.PythonImage(python_version="3.10.15").python_packages(
    "scikit-learn==1.4.2",
    "pandas",
    "numpy",
    "pydantic",
    "typing",
    "pycaret[models]==3.3.2",
)

with bentoml.importing():
    import numpy as np
    import pandas as pd
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any, Optional
    from bentoml.models import BentoModel
    import joblib
    import yaml
    from pycaret.regression import predict_model


# Define input schemas outside the service class
class HousingFeatures(BaseModel):
    # Original untransformed features
    Suburb: str = Field(..., description="Suburb name")
    Rooms: int = Field(..., description="Number of rooms")
    Type: str = Field(
        ..., description="Property type (House, Townhouse, Unit/Apartment)"
    )
    Method: str = Field(..., description="Sale method (PI, S, SA, SP, VB)")
    Seller: str = Field(..., description="Real estate agency")
    Distance: float = Field(..., description="Distance from CBD in kilometers")
    Bathroom: float = Field(..., description="Number of bathrooms")
    Car: int = Field(..., description="Number of car spaces")
    Landsize: float = Field(..., description="Land size in square meters")
    BuildingArea: Optional[float] = Field(
        None, description="Building area in square meters"
    )
    PropertyAge: Optional[int] = Field(None, description="Age of the property in years")
    Direction: Optional[str] = Field(None, description="Direction (N, S, E, W)")
    LandSizeNotOwned: Optional[bool] = Field(
        False, description="Flag indicating if land is not owned with the property"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "Suburb": "Reservoir",
                "Rooms": 3,
                "Type": "House",
                "Method": "S",
                "Seller": "Ray",
                "Distance": 11.2,
                "Bathroom": 1.0,
                "Car": 2,
                "Landsize": 556.0,
                "BuildingArea": 120.0,
                "PropertyAge": 50,
                "Direction": "N",
                "LandSizeNotOwned": False,
            }
        },
    }


class BatchHousingFeatures(BaseModel):
    features: List[HousingFeatures]

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    {
                        "Suburb": "Reservoir",
                        "Rooms": 3,
                        "Type": "House",
                        "Method": "S",
                        "Seller": "Ray",
                        "Distance": 11.2,
                        "Bathroom": 1.0,
                        "Car": 2,
                        "Landsize": 556.0,
                        "BuildingArea": 120.0,
                        "PropertyAge": 50,
                        "Direction": "N",
                        "LandSizeNotOwned": False,
                    },
                    {
                        "Suburb": "Richmond",
                        "Rooms": 2,
                        "Type": "Unit/Apartment",
                        "Method": "SA",
                        "Seller": "Jellis",
                        "Distance": 3.5,
                        "Bathroom": 1.0,
                        "Car": 1,
                        "Landsize": 0.0,
                        "BuildingArea": 85.0,
                        "PropertyAge": 15,
                        "Direction": "E",
                        "LandSizeNotOwned": True,
                    },
                ]
            }
        }
    }


# Get the model from BentoML model store
model_ref = bentoml.sklearn.get("housing_price_model:latest")


@bentoml.service(
    name="housing_price_prediction", image=bento_image, resources={"cpu": "1"}
)
class HousingPriceService:
    model_ref = BentoModel("housing_price_model:latest")

    def __init__(self):
        # Directly load the model file using joblib
        self.model = joblib.load(self.model_ref.path_of("saved_model.pkl"))

        # Get lambda values from model's custom objects
        self.custom_objects = joblib.load(self.model_ref.path_of("custom_objects.pkl"))

        # Load model.yaml for additional metadata
        with open(self.model_ref.path_of("model.yaml"), "r") as f:
            self.model_yaml = yaml.safe_load(f)

        self.boxcox_store = self.custom_objects.get("boxcox_store", {})

        # Import PyCaret's predict_model function

        self.predict_model = predict_model

        print(f"Loaded boxcox_store: {self.boxcox_store}")

    @bentoml.api
    def predict(self, housing_features: HousingFeatures) -> Dict[str, Any]:
        """
        Predict housing price based on input features.
        Returns the price in original scale.
        """
        # Convert input to dictionary
        input_dict = housing_features.model_dump()

        # Transform input
        transformed_input = self.preprocess_for_prediction(pd.DataFrame([input_dict]))

        # Make prediction using PyCaret's predict_model
        prediction_df = self.predict_model(self.model, data=transformed_input)

        # Extract the transformed prediction
        transformed_price = prediction_df["prediction_label"].values[0]

        # Convert price back to original scale
        original_price = self.inverse_transform_price(transformed_price)

        return {
            "prediction": float(original_price),
            "prediction_transformed": float(transformed_price),
            "currency": "AUD",
            "input_features": input_dict,
            "model_info": {
                "model_type": self.custom_objects.get("model_type", "Unknown"),
                "features_used": len(transformed_input.columns),
            },
        }

    @bentoml.api
    def batch_predict(self, batch_features: BatchHousingFeatures) -> Dict[str, Any]:
        """
        Predict housing prices for multiple inputs.
        Returns prices in original scale.
        """
        # Convert input to list of dictionaries
        input_dicts = [f.model_dump() for f in batch_features.features]

        # Transform inputs
        transformed_inputs = self.preprocess_for_prediction(pd.DataFrame([input_dicts]))

        # Make predictions using PyCaret's predict_model
        prediction_df = self.predict_model(self.model, data=transformed_inputs)

        # Extract the transformed predictions
        transformed_prices = prediction_df["prediction_label"].values

        # Convert prices back to original scale
        original_prices = [
            float(self.inverse_transform_price(p)) for p in transformed_prices
        ]

        return {
            "predictions": original_prices,
            "predictions_transformed": transformed_inputs,
            "currency": "AUD",
            "count": len(original_prices),
            "model_info": {
                "model_type": self.custom_objects.get("model_type", "Unknown"),
                "features_used": len(transformed_inputs.columns),
            },
        }

    def preprocess_for_prediction(self, df):
        """
        Apply all necessary preprocessing steps to prepare data for prediction.

        Args:
            df: DataFrame with raw input features

        Returns:
            DataFrame with all features required by the model
        """
        processed = df.copy()
        print("before processing", processed.to_dict())

        # 1. Property Type encoding
        if "Type" in processed.columns:
            # Map text property types to codes if needed
            type_mapping = {"House": "h", "Unit/Apartment": "u", "Townhouse": "t"}
            if not processed["Type"].isin(["h", "u", "t"]).all():
                processed["Type"] = processed["Type"].map(
                    lambda x: type_mapping.get(x, x)
                )

            # Create one-hot encoded features
            processed["PropType_House"] = (processed["Type"] == "h").astype(int)
            processed["PropType_Townhouse"] = (processed["Type"] == "t").astype(int)
            processed["PropType_Unit/Apartment"] = (processed["Type"] == "u").astype(
                int
            )
            processed = processed.drop("Type", axis=1)

        # 2. Method encoding
        if "Method" in processed.columns:
            # Create one-hot encoded Method features
            method_columns = [
                "Method_PI",
                "Method_S",
                "Method_SA",
                "Method_SP",
                "Method_VB",
            ]
            for col in method_columns:
                method_code = col.split("_")[1]
                processed[col] = (processed["Method"] == method_code).astype(int)
            processed = processed.drop("Method", axis=1)

        # 3. Suburb encoding
        if "Suburb" in processed.columns:
            # Use the suburb_to_rank_dict from the custom objects
            suburb_ranks = self.custom_objects.get("suburb_to_rank_dict", {})
            processed["Suburb_PriceRank"] = processed["Suburb"].map(
                suburb_ranks, na_action="ignore"
            )
            # Fill missing values with median rank
            if processed["Suburb_PriceRank"].isna().any():
                median_rank = np.median(list(suburb_ranks.values()))
                processed["Suburb_PriceRank"] = processed["Suburb_PriceRank"].fillna(
                    median_rank
                )
            processed = processed.drop("Suburb", axis=1)

        # 4. Seller encoding
        if "Seller" in processed.columns:
            # Create one-hot encoded Seller features
            common_sellers = [
                "Barry",
                "Biggin",
                "Brad",
                "Buxton",
                "Fletchers",
                "Gary",
                "Greg",
                "Harcourts",
                "Hodges",
                "Jas",
                "Jellis",
                "Kay",
                "Love",
                "Marshall",
                "McGrath",
                "Miles",
                "Nelson",
                "Noel",
                "RT",
                "Raine",
                "Ray",
                "Stockdale",
                "Sweeney",
                "Village",
                "Williams",
                "Woodards",
                "YPA",
                "hockingstuart",
            ]

            # Initialize all seller columns to 0
            for seller in common_sellers:
                processed[f"Seller_{seller}"] = 0
            processed["Seller_Other"] = 0

            # Set the appropriate column to 1 or "Other" if not in common sellers
            for idx, seller in enumerate(processed["Seller"]):
                if seller in common_sellers:
                    processed.loc[idx, f"Seller_{seller}"] = 1
                else:
                    processed.loc[idx, "Seller_Other"] = 1

            processed = processed.drop("Seller", axis=1)

        # 5. Direction features if needed
        if "Direction" in processed.columns:
            direction_cols = [
                "Direction_N",
                "Direction_S",
                "Direction_E",
                "Direction_W",
            ]
            for dir_col in direction_cols:
                dir_code = dir_col.split("_")[1]
                processed[dir_col] = (processed["Direction"] == dir_code).astype(int)
            processed = processed.drop("Direction", axis=1)

        # 6. Numerical transformations
        # Apply Box-Cox transformations to numeric features
        if "Landsize" in processed.columns:
            processed["Landsize_Transformed"] = self.box_cox_transform(
                processed["Landsize"],
                self.boxcox_store["landsize_lambda"],
                self.boxcox_store.get("landsize_offset", 0),
            )

        if "BuildingArea" in processed.columns:
            processed["BuildingArea_Transformed"] = self.box_cox_transform(
                processed["BuildingArea"],
                self.boxcox_store["building_area_lambda"],
                self.boxcox_store.get("building_area_offset", 0),
            )

        if "Distance" in processed.columns:
            processed["Distance_Transformed"] = self.box_cox_transform(
                processed["Distance"],
                self.boxcox_store["distance_lambda"],
                self.boxcox_store.get("distance_offset", 0),
            )

        if "Rooms" in processed.columns:
            processed["Rooms_Transformed"] = self.box_cox_transform(
                processed["Rooms"],
                self.boxcox_store["rooms_lambda"],
                self.boxcox_store.get("rooms_offset", 0),
            )

        if "Bathroom" in processed.columns:
            processed["Bathroom_Transformed"] = self.box_cox_transform(
                processed["Bathroom"],
                self.boxcox_store["bathroom_lambda"],
                self.boxcox_store.get("bathroom_offset", 0),
            )

        if "Car" in processed.columns:
            processed["Car_Transformed"] = self.box_cox_transform(
                processed["Car"],
                self.boxcox_store["car_lambda"],
                self.boxcox_store.get("car_offset", 0),
            )

        if "PropertyAge" in processed.columns:
            processed["PropertyAge_Transformed"] = self.box_cox_transform(
                processed["PropertyAge"],
                self.boxcox_store["propertyage_lambda"],
                self.boxcox_store.get("propertyage_offset", 0),
            )

        # Convert boolean values to integers
        bool_columns = processed.select_dtypes(include=["bool"]).columns
        for col in bool_columns:
            processed[col] = processed[col].astype(int)

        # 7. Add any missing columns required by the model with default values (0)
        required_columns = [
            "PropType_House",
            "PropType_Townhouse",
            "PropType_Unit/Apartment",
            "Method_PI",
            "Method_S",
            "Method_SA",
            "Method_SP",
            "Method_VB",
            "Suburb_PriceRank",
            "Seller_Barry",
            "Seller_Biggin",
            "Seller_Brad",
            "Seller_Buxton",
            "Seller_Fletchers",
            "Seller_Gary",
            "Seller_Greg",
            "Seller_Harcourts",
            "Seller_Hodges",
            "Seller_Jas",
            "Seller_Jellis",
            "Seller_Kay",
            "Seller_Love",
            "Seller_Marshall",
            "Seller_McGrath",
            "Seller_Miles",
            "Seller_Nelson",
            "Seller_Noel",
            "Seller_Other",
            "Seller_RT",
            "Seller_Raine",
            "Seller_Ray",
            "Seller_Stockdale",
            "Seller_Sweeney",
            "Seller_Village",
            "Seller_Williams",
            "Seller_Woodards",
            "Seller_YPA",
            "Seller_hockingstuart",
            "Landsize_Transformed",
            "BuildingArea_Transformed",
            "Distance_Transformed",
            "Rooms_Transformed",
            "Bathroom_Transformed",
            "Car_Transformed",
            "PropertyAge_Transformed",
            "Direction_N",
            "Direction_S",
            "Direction_E",
            "Direction_W",
            "LandSizeNotOwned",
        ]

        for col in required_columns:
            if col not in processed.columns:
                processed[col] = 0

        print("after processing", processed.to_dict())
        return processed

    def box_cox_transform(self, values, lambda_val, offset=0):
        """Apply Box-Cox transformation to a series of values"""
        values = pd.to_numeric(values, errors="coerce")
        # Handle NaNs
        values = values.fillna(values.median())

        # Apply offset if needed
        values_offset = values + offset

        # Apply transformation
        if abs(lambda_val) < 1e-10:  # lambda is close to zero
            return np.log(values_offset)
        else:
            return ((values_offset**lambda_val) - 1) / lambda_val

    def inverse_transform_price(self, transformed_value):
        """
        Inverse-transform a price value using the stored PowerTransformer.

        Args:
            transformed_value: Transformed price value

        Returns:
            Original price value
        """
        # Get the offset
        offset = self.boxcox_store.get("price_offset", 0)

        # Get the transformer from boxcox_store
        if "price_transformer" in self.boxcox_store:
            pt = self.boxcox_store["price_transformer"]

            # Reshape for scikit-learn
            value_reshaped = np.array([transformed_value]).reshape(-1, 1)

            # Use the transformer's built-in inverse_transform method
            original_with_offset = pt.inverse_transform(value_reshaped)[0][0]

            # Remove the offset
            original_price = original_with_offset - offset

            return original_price
        else:
            # Fallback to manual implementation if transformer isn't available
            lambda_val = self.boxcox_store["price_lambda"]

            if abs(lambda_val) < 1e-10:  # lambda is close to zero
                x_original = np.exp(transformed_value)
            else:
                x_original = np.power(
                    lambda_val * transformed_value + 1, 1 / lambda_val
                )

            return x_original - offset

    @bentoml.api
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the model.
        """
        return {
            "name": "Housing Price Prediction Model",
            "model_type": self.custom_objects.get("model_type", "Unknown"),
            "features": self.custom_objects.get("feature_names", []),
            "target": self.custom_objects.get("target_name", "Price"),
            "boxcox_store": self.boxcox_store,
            "metadata": self.model_yaml,  # Include the YAML metadata
            "description": "This model predicts housing prices in Melbourne based on property features.",
            "input_example": {
                "Suburb": "Reservoir",
                "Rooms": 3,
                "Type": "House",
                "Method": "S",
                "Seller": "Ray",
                "Distance": 11.2,
                "Bathroom": 1.0,
                "Car": 2,
                "Landsize": 556.0,
                "BuildingArea": 120.0,
                "PropertyAge": 50,
                "Direction": "N",
                "LandSizeNotOwned": False,  # Add to example
            },
        }
