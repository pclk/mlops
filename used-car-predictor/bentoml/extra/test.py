import requests

# Endpoint for BentoML API
BENTOML_ENDPOINT = "https://car-price-predictor-m9l0-1531a252.mt-guc1.bentoml.ai/predict"

def test_predict_car_price():
    # Test input matching CarInput schema
    input_data = {
        "input": [
            {
                "brand_model": "Maruti Swift VDI",
                "location": "Mumbai",
                "year": 2018,
                "kilometers_driven": 30000,
                "fuel_type": "Petrol",
                "transmission": "Manual",
                "owner_type": "First",
                "mileage": 18.5,
                "power": 82.0,
                "seats": 5
            },
            {
                "brand_model": "Maruti Swift VDI",
                "location": "Mumbai",
                "year": 2017,
                "kilometers_driven": 30000,
                "fuel_type": "Petrol",
                "transmission": "Manual",
                "owner_type": "First",
                "mileage": 18.5,
                "power": 90.0,
                "seats": 5
            }
        ]
    }

    # Send POST request
    response = requests.post(BENTOML_ENDPOINT, json=input_data)

    # Check response status
    assert response.status_code == 200, f"Failed with status code {response.status_code}"

    # Parse JSON response
    predictions = response.json()

    # Expected range of predictions (allowing slight variance)
    expected_predictions = [579966.45, 584712.52]
    tolerance = 10000  # Allow ±10,000 INR difference

    assert len(predictions) == 2, "Expected 2 predictions, got a different count."
    
    # Validate each prediction within range
    for i, pred in enumerate(predictions):
        assert "predicted_price" in pred, "Missing 'predicted_price' key in response."
        predicted_price = pred["predicted_price"]
        expected_price = expected_predictions[i]

        assert (
            expected_price - tolerance <= predicted_price <= expected_price + tolerance
        ), f"Prediction {i+1} out of expected range: {predicted_price} not within ±{tolerance} of {expected_price}"

    print("✅ All predictions are within the expected range.")

test_predict_car_price()