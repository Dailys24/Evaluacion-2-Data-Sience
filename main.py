# PROYECTO FINAL DATA SCIENCE: Modelo de Predicción Turistas 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression     
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Carga y Limpieza de Datos

# 1.1. Carga y procesamiento de datos

file_name = 'Informacion.csv' 

try:
    df = pd.read_csv(file_name, dtype=str)
    print(f"Archivo '{file_name}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_name}'.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

if 'Dolar' in df.columns:
    df.rename(columns={'Dolar': 'Tipo_Cambio_Promedio'}, inplace=True)


def clean_decimal(x):
    if pd.isna(x): return np.nan
    x = str(x)
    x = x.replace('.', '') 
    x = x.replace(',', '.') 
    try:
        return float(x)
    except:
        return np.nan

def clean_integer(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('.', '') 
    try:
        return int(x)
    except:
        return np.nan

# 1.2. Limpieza de Datos en CSV:
if 'Tipo_Cambio_Promedio' in df.columns: df['Tipo_Cambio_Promedio'] = df['Tipo_Cambio_Promedio'].apply(clean_decimal)
if 'PIB' in df.columns: df['PIB'] = df['PIB'].apply(clean_decimal)

cols_enteros = ['Robos_violentos', 'Delitos_contra_propiedad_no_violentos', 'Poblacion_RM']
for col in cols_enteros:
    if col in df.columns:
        df[col] = df[col].apply(clean_integer)

print("Variables económicas y delictuales limpiadas.")

col_turistas = 'cantidad_de_Turistas_Extranjeros'
if col_turistas in df.columns:
    df[col_turistas] = df[col_turistas].apply(clean_integer) 

df['Año'] = pd.to_numeric(df['Año'])

mapa_meses = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
              'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
df['Mes_Num'] = df['Mes'].map(mapa_meses)

df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
df = df.sort_values(by='Fecha').set_index('Fecha')

df['Lag_1'] = df[col_turistas].shift(1)
df['Lag_2'] = df[col_turistas].shift(2) 
df['Lag_3'] = df[col_turistas].shift(3) 
df['Lag_12'] = df[col_turistas].shift(12)

df_historico = df.dropna()
df_futuro_a_predecir = df[pd.isna(df[col_turistas]) & (df.index.year >= 2025)]

print(f"\nDatos Históricos Completos: {len(df_historico)} filas.")
print(f"Meses a Predecir (2025): {len(df_futuro_a_predecir)} filas.")

# 2. Evaluación de Modelos

print("\n" + "="*60)
print("EVALUACIÓN DE MODELOS")
print("="*60)

df_eval = df_historico.copy() 
media_turistas = df_eval[col_turistas].mean()
df_eval['Temporada_Alta'] = (df_eval[col_turistas] > media_turistas).astype(int)

# 3. Definición de Variables

features = [
    'Mes_Num', 'Año', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12', 
    'Tipo_Cambio_Promedio', 'PIB', 
    'Robos_violentos', 'Delitos_contra_propiedad_no_violentos',
    'Poblacion_RM'
]

X = df_eval[features]
y_regresion = df_eval[col_turistas]
y_clasificacion = df_eval['Temporada_Alta']

test_size = 0.3 
split_index = int(len(df_eval) * (1 - test_size))

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_reg, y_test_reg = y_regresion.iloc[:split_index], y_regresion.iloc[split_index:]
y_train_clas, y_test_clas = y_clasificacion.iloc[:split_index], y_clasificacion.iloc[split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Datos de entrenamiento: {len(X_train)} filas")
print(f"Datos de prueba: {len(X_test)} filas")


# 4. Implementación de Modelos

print("\nModelo 1: Regresión Lineal")
modelo_reg_lineal = LinearRegression()
modelo_reg_lineal.fit(X_train, y_train_reg)
pred_reg_lineal = modelo_reg_lineal.predict(X_test)
r2_lineal = r2_score(y_test_reg, pred_reg_lineal)
print(f"R^2 Regresión Lineal: {r2_lineal:.1f}")


print("\n[Modelo 2: Regresión Logística (Clasificación)]")
modelo_reg_log = LogisticRegression(random_state=42)
modelo_reg_log.fit(X_train_scaled, y_train_clas)
pred_reg_log = modelo_reg_log.predict(X_test_scaled)
acc_log = accuracy_score(y_test_clas, pred_reg_log)
print(f"Precisión: {acc_log:.1f}")


print("\nModelo 3: Random Forest")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train_clas)
pred_rf = modelo_rf.predict(X_test)
acc_rf = accuracy_score(y_test_clas, pred_rf)
print(f"Precisión: {acc_rf:.1f}")

importancias = modelo_rf.feature_importances_
df_importancias = pd.DataFrame({'Variable': features, 'Importancia': importancias})
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(x='Importancia', y='Variable', data=df_importancias)
plt.title('Importancia de Variables (Random Forest)')
plt.show()


print("\nModelo 4: Red Neuronal Profunda:")

# 4.1. Arquitectura de Red Neuronal
modelo_keras = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.2), 
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

modelo_keras.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])

# 4.2. Entrenamiento de Red Neuronal
print("Entrenando IA")
history = modelo_keras.fit(
    X_train_scaled, y_train_reg,
    epochs=100,           
    batch_size=16,        
    validation_split=0.2,
    verbose=0            
)

# 4.3. Evaluación de Red Neuronal
pred_nn_keras = modelo_keras.predict(X_test_scaled).flatten()
r2_keras = r2_score(y_test_reg, pred_nn_keras)
print(f"R^2 Red Neuronal Keras: {r2_keras:.1f}")

# 4.4. Curva de Aprendizaje Red Neuronal
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Error Entrenamiento (Loss)')
plt.plot(history.history['val_loss'], label='Error Validación (Val Loss)')
plt.title('Curva de Aprendizaje - Red Neuronal Keras')
plt.xlabel('Épocas')
plt.ylabel('Error (MSE)')
plt.legend()
plt.grid(True)
plt.show() 

# 4.5. Gráfico Real vs Predicción Red Neuronal
plt.figure(figsize=(10, 5))
plt.plot(y_regresion.index, y_regresion, label='Datos Reales', color='black', lw=2)
plt.plot(y_test_reg.index, pred_nn_keras, label='Predicción IA', color='green', linestyle='--')
plt.title('Red Neuronal Keras: Realidad vs Predicción')
plt.legend()
plt.show()

# 5. Resumen de Modelos

print("\n--- RESUMEN DE MODELOS ---")
print(f"R^2 Regresión Lineal : {r2_lineal:.1f}")
print(f"R^2 Red Neuronal (IA): {r2_keras:.1f}")
print(f"Precisión RF Clasif  : {acc_rf:.1f}")

# 6. Predicción Final 2025

print("\n" + "="*60)
print("INICIANDO PROYECCIÓN PARA EL AÑO 2025...")
print("="*60)

X_historico_total = df_historico[features]
y_historico_total = df_historico[col_turistas]

modelo_final = LinearRegression()
modelo_final.fit(X_historico_total, y_historico_total)
print("Modelo final ajustado.")

print("\nCalculando proyecciones...")
df_prediccion = df.copy()

cols_a_rellenar = ['Tipo_Cambio_Promedio', 'PIB', 'Robos_violentos', 'Delitos_contra_propiedad_no_violentos', 'Poblacion_RM']
cols_presentes = [c for c in cols_a_rellenar if c in df_prediccion.columns]
df_prediccion[cols_presentes] = df_prediccion[cols_presentes].fillna(method='ffill')

for fecha_a_predecir in df_futuro_a_predecir.index:
    df_prediccion['Lag_1'] = df_prediccion[col_turistas].shift(1)
    df_prediccion['Lag_2'] = df_prediccion[col_turistas].shift(2)
    df_prediccion['Lag_3'] = df_prediccion[col_turistas].shift(3)
    df_prediccion['Lag_12'] = df_prediccion[col_turistas].shift(12)
    
    X_actual = df_prediccion.loc[[fecha_a_predecir]][features]
    
    prediccion_turistas = modelo_final.predict(X_actual)
    
    df_prediccion.loc[fecha_a_predecir, col_turistas] = int(prediccion_turistas[0])
    print(f"-> {fecha_a_predecir.strftime('%B %Y')}: {int(prediccion_turistas[0])} turistas")

# 7. Gráfico Final
print("\nGRÁFICO FINAL DE PROYECCIÓN")
plt.figure(figsize=(12, 6))
plt.plot(df_historico.index, df_historico[col_turistas], label='Histórico', color='blue', lw=2)
plt.plot(df_prediccion.loc[df_futuro_a_predecir.index].index, 
         df_prediccion.loc[df_futuro_a_predecir.index][col_turistas], 
         label='Proyección 2025', color='red', linestyle='--', marker='o')

plt.title('Proyección de Turistas Extranjeros 2025')
plt.ylabel('Cantidad de Turistas')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\nGracias por utilizar el codigo")


