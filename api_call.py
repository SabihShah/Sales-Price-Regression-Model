import requests

url = 'http://127.0.0.1:8000/predict'
data = {
    "Lbmo_Material": 32413.235,
    "Total_Factura": 344874.32,
    "Costo_Uni_MP": 9.08,
    "Costo_Uni_FA": 0.44,
    "Costo_MP": 294454.36,
    "Costo_FA": 294454.36,
    "Costo_Total": 308573.08,
    "margen_Venta": 36301.24,
    "Contrato": "Contrato Regular"
}

response = requests.post(url, json=data)
print(response.json())
