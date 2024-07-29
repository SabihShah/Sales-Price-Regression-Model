import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle



data = pd.read_csv('sales_data.csv')
# data.head()

value_to_drop = 'Venta de Muestra'
data = data[data['Motivo Pedido'] != value_to_drop]
# data.value_counts('Motivo Pedido')

new_data = data.drop(columns=['Número Sap', 'Centro', 'Organización de ventas', 'Denominacion', 'N° Client', 'Clase Factura', 'Canal de Venta',
                      'Documento', 'Descripcion Contrato', 'Nº Material', 'Fecha Factura', 'anonymized_customer', 'anonymized_material',
                      'Planta', 'Región', 'Moneda', 'Motivo Pedido'])

new_data['Base Fixed'] = new_data['Precio Unitario Venta  US$/Lb']


category_counts = data['Motivo Pedido'].value_counts()
categories = category_counts.index
values = category_counts.values

colors = ['blue', 'green']
fig, ax = plt.subplots()
bars = ax.bar(categories, values, color=colors)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')


# Add titles and labels
ax.set_title('# of Occurrences')
ax.set_xlabel('Categories')
ax.set_ylabel('Number of Occurrences')
plt.show()

# Improve layout
plt.xticks(rotation=45)
plt.tight_layout()

fig, ax = plt.subplots()


ax.scatter(new_data['Lbmo Material'], new_data['Precio Unitario Venta  US$/Lb'], s=100, edgecolor='black', alpha=0.75)
ax.set_yscale('log')
ax.set_title('Lbmao Material v/s Precio Unitario Venta  US$/Lb')
ax.set_xlabel('Lbmao Material')
ax.set_ylabel('Precio Unitario Venta US$/Lb')
plt.show()

new_data['Surcharge'] = new_data['Surcharge'].fillna(new_data['Surcharge'].mean())
new_data['Factor BQM'] = new_data['Factor BQM'].fillna(new_data['Factor BQM'].mean())
new_data['Factor Precio Base'] = new_data['Factor Precio Base'].fillna(new_data['Factor Precio Base'].mean())


z_score_scaler = StandardScaler()

data_scaled = pd.DataFrame(z_score_scaler.fit_transform(new_data), 
                           columns=['Lbmo Material', 'Total Factura', 'Precio Unitario Venta US$/Lb', 'Costo Uni. MP', 'Costo Uni. FA', 'Costo MP',
                                    'Costo FA',	'Costo Total', 'margen Venta', 'Surcharge',	'Base Fixed', 'Factor BQM',	'Factor Precio Base'])

x = new_data[['Lbmo Material', 'Total Factura', 'Costo Uni. MP', 'Costo Uni. FA', 'Costo MP', 'Costo FA', 'Costo Total', 'margen Venta']]
y = new_data[['Precio Unitario Venta  US$/Lb', 'Surcharge', 'Factor BQM']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


numerical_features = ['Lbmo Material', 'Total Factura', 'Costo Uni. MP', 'Costo Uni. FA', 'Costo MP', 'Costo FA', 'Costo Total', 'margen Venta']
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])



model = MultiOutputRegressor(RandomForestRegressor(random_state=42))

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

trained_model = pipeline.fit(X_train, y_train)

y_pred = trained_model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
rmse = mean_squared_error(y_test, y_pred, multioutput='raw_values', squared=False)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)



# Prediction

def predict(Lbmo_Material, Total_Factura, Costo_Uni_MP, Costo_Uni_FA, Costo_MP, Costo_FA, Costo_Total, margen_Venta, 
               Contrato='Contrato Spot', model=trained_model):
    
    Custom_inputs = {'Lbmo Material': Lbmo_Material, 
                    'Total Factura':Total_Factura, 
                    'Costo Uni. MP':Costo_Uni_MP, 
                    'Costo Uni. FA':Costo_Uni_FA, 
                    'Costo MP':Costo_MP, 
                    'Costo FA':Costo_FA, 
                    'Costo Total':Costo_Total, 
                    'margen Venta':margen_Venta}


    inputs_df = pd.DataFrame(Custom_inputs)
    
    prediction = model.predict(inputs_df)
    
    if Contrato == 'Contrato Spot':
        sales_price = prediction[0][0]
        print('Sales Price:', sales_price)
    
    elif Contrato == 'Contrato Regular':
        sales_price = (prediction[0][0]*prediction[0][2])+prediction[0][1]
        print('Sales Price:', sales_price)
        
    else:
        print('Wrong parameters!')
        predict(Lbmo_Material, Total_Factura, Costo_Uni_MP, Costo_Uni_FA, Costo_MP, Costo_FA, Costo_Total, margen_Venta, 
               Contrato='Contrato Spot', model=trained_model)
    
    return sales_price


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)
# prediction = predict([32413.235], [344874.32], [9.08], [0.44], [294454.36], [294454.36], [308573.08], [36301.24], Contrato='Contrato Regular')