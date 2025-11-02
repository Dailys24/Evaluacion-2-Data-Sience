#Librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Modelos
from sklearn.linear_model import LinearRegression      #Modelo 1 (Regresión)
from sklearn.linear_model import LogisticRegression    #Modelo 2 (Clasificación)
from sklearn.ensemble import RandomForestClassifier  #Modelo 3 (Árbol/RF)
# Se eliminó la Red Neuronal de los imports

#Métricas de Evaluación
from sklearn.metrics import r2_score, mean_squared_error   # Para Regresión
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Para Clasificación

# --- 1. CARGA Y LIMPIEZA DE DATOS ---

#Carga y preprocesamiento de datos
file_name = 'Cantidad de turistas.csv' 

try:
    df_original = pd.read_csv(file_name)
    print(f"Archivo '{file_name}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_name}'.")
    exit()
except Exception as e:
    #Si falla UTF-8, intentar con 'latin1'
    try:
        df_original = pd.read_csv(file_name, encoding='latin1')
        print(f"Archivo '{file_name}' cargado con 'latin1'.")
    except Exception as e2:
        print(f"Error al leer el archivo. Revisa el formato. Error: {e2}")
        exit()

#Ingeniería de Características y Limpieza
df = df_original.copy()

#Limpiar nuevas columnas (Dólar y PIB)
if 'Tipo_Cambio_Promedio' in df.columns and df['Tipo_Cambio_Promedio'].dtype == 'object':
    df['Tipo_Cambio_Promedio'] = df['Tipo_Cambio_Promedio'].str.replace('.', '', regex=False)
    df['Tipo_Cambio_Promedio'] = df['Tipo_Cambio_Promedio'].str.replace(',', '.', regex=False)
    df['Tipo_Cambio_Promedio'] = pd.to_numeric(df['Tipo_Cambio_Promedio'], errors='coerce')
    
if 'PIB' in df.columns and df['PIB'].dtype == 'object':
    df['PIB'] = df['PIB'].str.replace('.', '', regex=False)
    df['PIB'] = df['PIB'].str.replace(',', '.', regex=False)
    df['PIB'] = pd.to_numeric(df['PIB'], errors='coerce')


cols_turistas = [col for col in df.columns if 'Turista' in col or 'turista' in col]
columna_turistas = cols_turistas[0] if cols_turistas else 'cantidad de Turistas Extranjeros'

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

#Crear variables desfasadas (Lags)
df['Lag_1'] = df[columna_turistas].shift(1)
df['Lag_12'] = df[columna_turistas].shift(12)

#Separar datos históricos y futuros
#df_historico son los datos que SÍ están completos (donde no hay NINGÚN NaN)
df_historico = df.dropna()

#df_futuro son las filas a predecir (donde 'cantidad de Turistas Extranjeros' está NaN)
df_futuro_a_predecir = df[pd.isna(df[columna_turistas]) & (df.index.year >= 2025)]

print(f"\nFilas ANTES de limpiar NaNs (Datos 2021-2025): {len(df)} (60 filas)")
print(f"Filas DESPUÉS de limpiar NaNs (Datos 2022-Mar 2025): {len(df_historico)} (39 filas)")

#PARTE 1: Evaluacion de modelos
print("\n\n" + "="*60)
print("INICIANDO PARTE 1: EVALUACIÓN DE MODELOS (RÚBRICA)")
print("="*60)

#Usamos una copia para no alterar los datos históricos
df_eval = df_historico.copy() 

#Crear variable objetivo de clasificación
media_turistas = df_eval[columna_turistas].mean()
df_eval['Temporada_Alta'] = (df_eval[columna_turistas] > media_turistas).astype(int)

print("\nVista previa de los datos de EVALUACIÓN (incluye Dólar y PIB):")
print(df_eval.head())


#Division de Datos (X e y)

#Definimos nuestras variables predictoras (Features)
features = ['Mes_Num', 'Año', 'Lag_1', 'Lag_12', 'Tipo_Cambio_Promedio', 'PIB']
X = df_eval[features]

#Problema 1: Regresión
y_regresion = df_eval[columna_turistas]

#Problema 2: Clasificación
y_clasificacion = df_eval['Temporada_Alta']

#Dividimos los datos: 60% para entrenar, 40% para probar (sin mezclar)
test_size = 0.4
split_index = int(len(df_eval) * (1 - test_size))

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_reg, y_test_reg = y_regresion.iloc[:split_index], y_regresion.iloc[split_index:]
y_train_clas, y_test_clas = y_clasificacion.iloc[:split_index], y_clasificacion.iloc[split_index:]

#Escalado de Datos (región logística y red neuronal)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDatos de entrenamiento: {len(X_train)} filas")
print(f"Datos de prueba: {len(X_test)} filas")

#Implementación de Modelos
print("\nIniciando entrenamiento de modelos de evaluación...")

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
plt.plot(y_regresion.index, y_regresion, label='Real (2022-Mar 2025)', color='black', lw=2)
plt.plot(y_train_reg.index, pred_train_lineal, label='Ajuste (Train)', color='blue', linestyle='--')
plt.plot(y_test_reg.index, pred_reg_lineal, label='Predicción (Test)', color='red', linestyle=':')
plt.title('[GRÁFICO DE EVALUACIÓN 1] Regresión Lineal: Predicción Completa')
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
plt.title('[GRÁFICO DE EVALUACIÓN 2] Matriz de Confusión - Reg. Logística')
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
plt.title('[GRÁFICO DE EVALUACIÓN 3] Importancia de Variables - Random Forest')
plt.show()

#Comparacion y resultados
print("\n--- RESUMEN DE EVALUACIÓN (PARTE 1) ---")
print("\nProblema de Regresión (Predecir Cantidad):")
print(f"R^2 Regresión Lineal: {r2_lineal:.1f}")

print("\nProblema de Clasificación (Predecir Temporada Alta):")
print(f"Accuracy Reg. Logística: {acc_log:.1f}")
print(f"Accuracy Random Forest : {acc_rf:.1f}")

#Prediccion final
print("\n\n" + "="*60)
print("INICIANDO PARTE 2: PREDICCIÓN DE FINALES DE 2025")
print("="*60)

#Separamos X e y del set de entrenamiento histórico COMPLETO (las 39 filas)
X_historico_total = df_historico[features]
y_historico_total = df_historico[columna_turistas]

#Entrenamos nuestro modelo final con TODOS los datos históricos
modelo_final = LinearRegression()
modelo_final.fit(X_historico_total, y_historico_total)
print("Modelo de Regresión Lineal (FINAL) entrenado con éxito.")

print("\nINICIANDO PREDICCIÓN PARA FINALES DE 2025")

#Hacemos una copia del dataframe original (el de 60 filas)
df_prediccion = df.copy()

#Llenamos los valores NaN de Dólar y PIB que faltan usando el último valor conocido
df_prediccion['Tipo_Cambio_Promedio'].fillna(method='ffill', inplace=True)
df_prediccion['PIB'].fillna(method='ffill', inplace=True)
print("Valores NaN de Dólar y PIB rellenados para poder predecir.")

#Bucle para predecir mes a mes
for fecha_a_predecir in df_futuro_a_predecir.index:
    df_prediccion['Lag_1'] = df_prediccion[columna_turistas].shift(1)
    df_prediccion['Lag_12'] = df_prediccion[columna_turistas].shift(12)
    
    #Extraemos las "pistas" (features) para el mes que queremos predecir
    X_actual = df_prediccion.loc[[fecha_a_predecir]][features]
    
    #Hacemos la predicción
    prediccion_turistas = modelo_final.predict(X_actual)
    
    #Guardamos la predicción en la columna de turistas
    df_prediccion.loc[fecha_a_predecir, columna_turistas] = int(prediccion_turistas[0])
    print(f"Predicción para {fecha_a_predecir.strftime('%Y-%m')}: {int(prediccion_turistas[0])} turistas")

#Resultados finales
print("\n--- DATAFRAME FINAL CON PREDICCIONES (AÑO 2025) ---")
#Filtramos solo por el año 2025 para ver la tabla final
print(df_prediccion[df_prediccion['Año'] == 2025])

#Gráfico final mostrando datos reales y la predicción
print("\nGenerando gráfico de predicción final...")
plt.figure(figsize=(12, 6))
#Graficamos los datos históricos conocidos (2022-Mar 2025)
plt.plot(df_historico[columna_turistas], label='Datos Históricos (2022-Mar 2025)', color='blue', lw=2)
#Graficamos predicción (Abr-Dic 2025)
plt.plot(df_prediccion.loc[df_futuro_a_predecir.index][columna_turistas], label='Predicción (Abr-Dic 2025)', color='red', linestyle='--')
plt.title('[GRÁFICO FINAL] Predicción de Turistas para Finales de 2025')
plt.ylabel('Cantidad de Turistas Extranjeros')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- SCRIPT ÚNICO COMPLETADO ---")