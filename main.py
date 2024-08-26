# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pickle
# import numpy as np

# app = FastAPI()

# # Load Spot model and scaler
# with open('Spot_Model_and_scaler.pkl', 'rb') as file:
#     spot_objects = pickle.load(file)
#     Spot_model = spot_objects['model']
#     Spot_scaler = spot_objects['scaler']

# # Load Regular models and scalers
# with open('Regular_models_and_scalers.pkl', 'rb') as file:
#     regular_objects = pickle.load(file)
#     Regular_models = regular_objects['models']
#     Regular_scaler_targets = regular_objects['scaler_target']
#     Regular_scaler_features = regular_objects['scaler_features']

# # Pydantic model for input data
# class PredictionRequest(BaseModel):
#     contractor_type: str  # 'spot' or 'regular'
#     N_material: int
#     Lbmo_material: float
#     Total_Factura: float
#     Costo_Total: float
#     margen_venta: float

# # Spot contract prediction function
# def predict_spot(N_material, Lbmo_material, Total_Factura, Costo_Total, margen_venta, model=Spot_model, scaler=Spot_scaler):
#     if N_material not in [2020, 1000, 2010, 2000, 2130, 2070, 2210]:
#         raise ValueError("Error: The chosen 'Nº Material' is not recognized by the model.")
    
#     features = np.array([Lbmo_material, Total_Factura, Costo_Total, margen_venta]).reshape(1, -1)
#     scaled = scaler.transform(features)
#     inputs = np.hstack([[N_material], scaled[0]])
#     input_data = inputs.reshape(1, -1)
#     prediction = model.predict(input_data)
    
#     return prediction[0]

# # Regular contract prediction function
# def predict_regular(N_Material, Lbmo_material, Total_Factura, Costo_Total, margen_venta, models=Regular_models, scalers_targets=Regular_scaler_targets, scaler_features=Regular_scaler_features):
    
#     if N_Material not in [2020, 2000, 2010]:
#         raise ValueError("Error: The chosen 'Nº Material' is not recognized by the model.")
    
#     # Scale the features using the fitted scaler
#     features_to_scale = np.array([Lbmo_material, Total_Factura, Costo_Total, margen_venta]).reshape(1,-1)
#     scaled_features = scaler_features.transform(features_to_scale)
#     features = np.hstack([[N_Material], scaled_features[0]]).reshape(1,-1)
#     # print(features)
    
#     # Predict each component and unscale the predictions
#     targets = ['Precio Unitario Venta  US$/Lb', 'Surcharge', 'Factor BQM']
#     predictions = {}
#     for target in targets:
#         pred_scaled = models[target].predict(features)
#         pred_unscaled = scalers_targets[target].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
#         predictions[target] = pred_unscaled
    
#     # Calculate the sales price
#     precio = predictions['Precio Unitario Venta  US$/Lb']
#     bqm = predictions['Factor BQM']
#     surcharge = predictions['Surcharge']
    
#     sales_price = (precio * bqm) + surcharge
    
#     print('Precio:', precio)
#     print('Factor BQM:', bqm)
#     print('Surcharge:', surcharge)
#     print('Sales Price:', sales_price)
    
#     return precio, bqm, surcharge, sales_price

# # API endpoint to handle both contractor types
# @app.post("/predict/")
# async def get_prediction(request: PredictionRequest):
#     try:
#         if request.contractor_type == 'spot':
#             prediction = predict_spot(
#                 request.N_material,
#                 request.Lbmo_material,
#                 request.Total_Factura,
#                 request.Costo_Total,
#                 request.margen_venta
#             )
        
#         elif request.contractor_type == 'regular':
#             prediction = predict_regular(
#                 request.N_material,
#                 request.Lbmo_material,
#                 request.Total_Factura,
#                 request.Costo_Total,
#                 request.margen_venta
#             )
           
#         else:
#             raise HTTPException(status_code=400, detail="Invalid contractor_type specified")

#         return {"prediction": prediction}
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="An error occurred during prediction")

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000)



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pickle
import numpy as np

app = FastAPI()

# Load Spot model and scaler
with open('Spot_Model_and_scaler.pkl', 'rb') as file:
    spot_objects = pickle.load(file)
    Spot_model = spot_objects['model']
    Spot_scaler = spot_objects['scaler']

# Load Regular models and scalers
with open('Regular_models_and_scalers.pkl', 'rb') as file:
    regular_objects = pickle.load(file)
    Regular_models = regular_objects['models']
    Regular_scaler_targets = regular_objects['scaler_target']
    Regular_scaler_features = regular_objects['scaler_features']

# Pydantic model for input data
class Feature(BaseModel):
    N_material: int
    Lbmo_material: float
    Total_Factura: float
    Costo_Total: float
    margen_venta: float

class PredictionRequest(BaseModel):
    contractor_type: str  # 'spot' or 'regular'
    features: List[Feature]  # List of feature objects

# Spot contract batch prediction function
def predict_spot_batch(features: List[Dict[str, float]], model=Spot_model, scaler=Spot_scaler):
    sales_prices = []
    for feature in features:
        if feature['N_material'] not in [2020, 1000, 2010, 2000, 2130, 2070, 2210]:
            sales_prices.append("Error: The chosen 'Nº Material' is not recognized by the model.")
            continue
        
        feats = np.array([feature['Lbmo_material'], feature['Total_Factura'], feature['Costo_Total'], feature['margen_venta']]).reshape(1,-1)
        scaled = scaler.transform(feats)
        inputs = np.hstack([[feature['N_material']], scaled[0]])
        input_data = inputs.reshape(1,-1)
        prediction = model.predict(input_data)
        sales_prices.append(prediction[0])
    
    return sales_prices

# Regular contract batch prediction function
def predict_regular_batch(features: List[Dict[str, float]], models=Regular_models, scalers_targets=Regular_scaler_targets, scaler_features=Regular_scaler_features):
    sales_prices = []
    for feature in features:
        if feature['N_material'] not in [2020, 2000, 2010]:
            sales_prices.append("Error: The chosen 'Nº Material' is not recognized by the model.")
            continue
        
        features_to_scale = np.array([feature['Lbmo_material'], feature['Total_Factura'], feature['Costo_Total'], feature['margen_venta']]).reshape(1,-1)
        scaled_features = scaler_features.transform(features_to_scale)
        features_array = np.hstack([[feature['N_material']], scaled_features[0]]).reshape(1,-1)
        
        targets = ['Precio Unitario Venta  US$/Lb', 'Surcharge', 'Factor BQM']
        
        preds = {}
        for target in targets:
            pred_scaled = models[target].predict(features_array)
            pred_unscaled = scalers_targets[target].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            preds[target] = pred_unscaled
            
        precio = preds['Precio Unitario Venta  US$/Lb']
        bqm = preds['Factor BQM']
        surcharge = preds['Surcharge']
            
        sales_price = (precio * bqm) + surcharge
        
        sales_prices.append(sales_price)
        
    return sales_prices

# API endpoint to handle batch predictions for both contractor types
@app.post("/predict/")
async def get_batch_prediction(request: PredictionRequest):
    try:
        if request.contractor_type == 'spot':
            predictions = predict_spot_batch(
                [feature.dict() for feature in request.features]
            )
        
        elif request.contractor_type == 'regular':
            predictions = predict_regular_batch(
                [feature.dict() for feature in request.features]
            )
           
        else:
            raise HTTPException(status_code=400, detail="Invalid contractor_type specified")

        return {"predictions": predictions}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during prediction")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
