from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import uvicorn


with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class PredictRequest(BaseModel):
    Lbmo_Material: float
    Total_Factura: float
    Costo_Uni_MP: float
    Costo_Uni_FA: float
    Costo_MP: float
    Costo_FA: float
    Costo_Total: float
    margen_Venta: float
    Contrato: str = 'Contrato Spot'

@app.post("/predict")
def predict(request: PredictRequest):
    inputs_df = pd.DataFrame([{
        'Lbmo Material': request.Lbmo_Material,
        'Total Factura': request.Total_Factura,
        'Costo Uni. MP': request.Costo_Uni_MP,
        'Costo Uni. FA': request.Costo_Uni_FA,
        'Costo MP': request.Costo_MP,
        'Costo FA': request.Costo_FA,
        'Costo Total': request.Costo_Total,
        'margen Venta': request.margen_Venta
    }])

    prediction = model.predict(inputs_df)
    if request.Contrato == 'Contrato Spot':
        sales_price = prediction[0][0]
    elif request.Contrato == 'Contrato Regular':
        sales_price = (prediction[0][0] * prediction[0][2]) + prediction[0][1]
    else:
        return {"error": "Invalid contract type"}

    return {"Sales Price": sales_price}


# uvicorn main:app --reload