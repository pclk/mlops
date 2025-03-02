import bentoml

with bentoml.SyncHTTPClient(
    "https://housing-price-prediction-csdc-f774f0f9.mt-guc1.bentoml.ai"
) as client:
    result = client.predict(
        housing_features={
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
    )

print("Result:", result)
