#Librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Modelos
from sklearn.linear_model import LinearRegression    #Modelo 1 (Regresión)
from sklearn.linear_model import LogisticRegression  #Modelo 2 (Clasificación)
from sklearn.ensemble import RandomForestClassifier  #Modelo 3 (Árbol/RF)
from sklearn.neural_network import MLPRegressor      #Modelo 4 (Red Neuronal)

#Métricas de Evaluación
from sklearn.metrics import r2_score, mean_squared_error   # Para Regresión
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Para Clasificación

#Carga y preprocesamiento de datos
file_name = 'Cantidad de turistas.csv' 

try:
    df = pd.read_csv(file_name)
    print(f"Archivo '{file_name}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_name}'.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo. Revisa el formato. Error: {e}")
    exit()

#Ingeniería de Características y Limpieza

#Limpiar nuevas columnas (Dólar y PIB)
if df['Tipo_Cambio_Promedio'].dtype == 'object':
    df['Tipo_Cambio_Promedio'] = df['Tipo_Cambio_Promedio'].str.replace(',', '.').astype(float)
if df['PIB'].dtype == 'object':
    df['PIB'] = df['PIB'].str.replace(',', '').astype(float)
print("Columnas 'Tipo_Cambio_Promedio' y 'PIB' limpiadas y convertidas a número.")

#Convertir 'Mes' (texto) a número
mapa_meses = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}
if df['Mes'].dtype == 'object':
    df['Mes_Num'] = df['Mes'].map(mapa_meses)
else:
    df['Mes_Num'] = df['Mes']

#Crear una 'Fecha' real para usarla como índice
df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes_Num'].astype(str))
df = df.sort_values(by='Fecha')
df = df.set_index('Fecha')

#Este gráfico muestra datos originales 2021-2024, ANTES de eliminar 2021)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['cantidad de Turistas Extranjeros'], label='Turistas Reales (2021-2024)')
plt.title('Gráfico Exploratorio: Turistas 2021-2024 (Datos Crudos)')
plt.legend()
plt.show()

#Crear variables desfasadas (Lags)
df['Lag_1'] = df['cantidad de Turistas Extranjeros'].shift(1)
df['Lag_12'] = df['cantidad de Turistas Extranjeros'].shift(12) # Tu idea de "enero vs enero"

#Limpieza
print(f"\nFilas ANTES de limpiar NaNs (Datos 2021-2024): {len(df)} (48 filas)")
df = df.dropna() # Esto elimina todo el año 2021 (12 filas) por el Lag_12
print(f"Filas DESPUÉS de limpiar NaNs (Datos 2022-2024): {len(df)} (36 filas)")

#Crear variable objetivo de clasificación
media_turistas = df['cantidad de Turistas Extranjeros'].mean()
df['Temporada_Alta'] = (df['cantidad de Turistas Extranjeros'] > media_turistas).astype(int)

print("\nVista previa de los datos procesados (incluye Dólar y PIB):")
print(df.head()) # Debería empezar en 2022-01-01


#Division de Datos (X e y)

#Definimos nuestras variables predictoras (Features)
features = ['Mes_Num', 'Año', 'Lag_1', 'Lag_12', 'Tipo_Cambio_Promedio', 'PIB']
X = df[features]

#Problema 1: Regresión
#Predecir la cantidad numérica de turistas
y_regresion = df['cantidad de Turistas Extranjeros']

#Problema 2: Clasificación
#Predecir la categoría 'Temporada_Alta' (1 o 0)
y_clasificacion = df['Temporada_Alta']

#Dividimos los datos: 60% para entrenar, 40% para probar (sin mezclar)
test_size = 0.4 #Probar
split_index = int(len(df) * (1 - test_size))

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_reg, y_test_reg = y_regresion.iloc[:split_index], y_regresion.iloc[split_index:]
y_train_clas, y_test_clas = y_clasificacion.iloc[:split_index], y_clasificacion.iloc[split_index:]

#Escalado de Datos (región logística y red neuronal)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDatos de entrenamiento: {len(X_train)} filas (Aprox. 21)")
print(f"Datos de prueba: {len(X_test)} filas (Aprox. 15)")

#Implementación de Modelos
print("\n--- INICIANDO ENTRENAMIENTO DE MODELOS ---")

#Modelo 1: Regresión Lineal (Regresión)
print("\n[Modelo 1: Regresión Lineal]")
modelo_reg_lineal = LinearRegression()
modelo_reg_lineal.fit(X_train, y_train_reg)
pred_reg_lineal = modelo_reg_lineal.predict(X_test)
r2_lineal = r2_score(y_test_reg, pred_reg_lineal)
print(f"R^2 (Regresión Lineal): {r2_lineal:.1f}")

#Gráfico (Real vs Predicción)
pred_train_lineal = modelo_reg_lineal.predict(X_train)

plt.figure(figsize=(10, 5))
plt.plot(y_regresion.index, y_regresion, label='Real (2022-2024)', color='black', lw=2)
plt.plot(y_train_reg.index, pred_train_lineal, label='Ajuste (Train 2022-23)', color='blue', linestyle='--')
plt.plot(y_test_reg.index, pred_reg_lineal, label='Predicción (Test 2023-24)', color='red', linestyle=':')
plt.title('Regresión Lineal: Predicción Completa (2022-2024)')
plt.legend()
plt.show()

#Modelo 2: Regresion logistica (Problema de Clasificación)
print("\n[Modelo 2: Regresión Logística]")
modelo_reg_log = LogisticRegression(random_state=42)
modelo_reg_log.fit(X_train_scaled, y_train_clas)
pred_reg_log = modelo_reg_log.predict(X_test_scaled)
acc_log = accuracy_score(y_test_clas, pred_reg_log)
print(f"Accuracy (Reg. Logística): {acc_log:.1f}")

#Matriz de Confusión
print("Matriz de Confusión (Reg. Logística):")
cm_log = confusion_matrix(y_test_clas, pred_reg_log)

sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Reg. Logística')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()

#Modelo 3: Random forest (Problema de Clasificación)
print("\n[Modelo 3: Random Forest Classifier]")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train_clas)
pred_rf = modelo_rf.predict(X_test)
acc_rf = accuracy_score(y_test_clas, pred_rf)
print(f"Accuracy (Random Forest): {acc_rf:.1f}")

#Importancia de Variables
importancias = modelo_rf.feature_importances_
df_importancias = pd.DataFrame({'Variable': features, 'Importancia': importancias})
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)
print("\nImportancia de Variables (RF):")
print(df_importancias)

sns.barplot(x='Importancia', y='Variable', data=df_importancias)
plt.title('Importancia de Variables - Random Forest')
plt.show()

#Modelo 4: Red neuronal - MLP (Problema de Regresión)
print("\n[Modelo 4: Red Neuronal (MLP Regressor)]")
modelo_nn = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42, solver='lbfgs')
modelo_nn.fit(X_train_scaled, y_train_reg)
pred_nn = modelo_nn.predict(X_test_scaled)
r2_nn = r2_score(y_test_reg, pred_nn)
print(f"R^2 (Red Neuronal): {r2_nn:.1f}")

#Gráfico (Comparando todos los modelos de regresión)
pred_train_nn = modelo_nn.predict(X_train_scaled)
plt.figure(figsize=(10, 5))
plt.plot(y_regresion.index, y_regresion, label='Real (2022-2024)', color='black', lw=2)
plt.plot(y_train_reg.index, pred_train_nn, label='Ajuste (Train 2022-23)', color='blue', linestyle='--')
plt.plot(y_test_reg.index, pred_nn, label='Predicción (Test 2023-24)', color='red', linestyle=':')
plt.title('Red Neuronal: Predicción Completa (2022-2024)')
plt.legend()
plt.show()

#Comparacion y resultados
print("\n--- RESUMEN DE EVALUACIÓN ---")
print("\nProblema de Regresión (Predecir Cantidad):")
print(f"R^2 Regresión Lineal: {r2_lineal:.1f}")
print(f"R^2 Red Neuronal    : {r2_nn:.1f}")

print("\nProblema de Clasificación (Predecir Temporada Alta):")
print(f"Accuracy Reg. Logística: {acc_log:.4f}")
print(f"Accuracy Random Forest : {acc_rf:.4f}")

print("\n--- FIN DEL SCRIPT ---")

