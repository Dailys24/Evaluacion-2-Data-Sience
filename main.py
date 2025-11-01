#Librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Modelos
from sklearn.linear_model import LinearRegression      # Modelo 1 (Regresión)
from sklearn.linear_model import LogisticRegression    # Modelo 2 (Clasificación)
from sklearn.ensemble import RandomForestClassifier  # Modelo 3 (Árbol/RF)
from sklearn.neural_network import MLPRegressor        # Modelo 4 (Red Neuronal)

#Métricas de Evaluación
from sklearn.metrics import r2_score, mean_squared_error   # Para Regresión
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Para Clasificación

#Carga y preprocesamiento de datos
try:
    #Período: Enero 2022 a Marzo 2025 = 39 meses.
    fechas = pd.date_range(start='2022-01-01', periods=39, freq='MS')
    turistas = []
    base_turistas = 150000
    for m in fechas:
        #Estacionalidad
        estacionalidad = (np.sin((m.month - 1) * (np.pi / 6) - (np.pi / 2)) * -1) * 30000 + 50000
        ruido = np.random.randint(-10000, 10000)
        #Simulación de recuperación post-pandemia
        turistas.append(int(estacionalidad + ruido + base_turistas * (m.year - 2021) * 0.5))
            
    df = pd.DataFrame({'Fecha': fechas, 'Cantidad_Turistas': turistas})
    #Fin Datos Dummy

except FileNotFoundError:
    print("Error: No se encontró el archivo 'tu_dataset_turismo.csv'.")
    exit()

#Convertir la columna 'Fecha' a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(by='Fecha') #Asegurarse de que esté ordenado
df = df.set_index('Fecha') #Poner la fecha como índice

#Crear variables de estacionalidad
df['Mes'] = df.index.month
df['Anio'] = df.index.year

#Crear variables desfasadas (Lags)
#Lag_1: Turistas del mes anterior
df['Lag_1'] = df['Cantidad_Turistas'].shift(1)

#Limpieza
print(f"Filas ANTES de limpiar NaNs (Ene 2022 - Mar 2025): {len(df)} (39 filas)")
df = df.dropna()
print(f"Filas DESPUÉS de limpiar NaNs (Feb 2022 - Mar 2025): {len(df)} (38 filas)")

#Crear variable objetivo de clasificación
media_turistas = df['Cantidad_Turistas'].mean()
df['Temporada_Alta'] = (df['Cantidad_Turistas'] > media_turistas).astype(int)

print("\nVista previa de los datos procesados:")
print(df.head())

#Division de Datos (X e y)

#Nuestras variables predictoras.
features = ['Mes', 'Anio', 'Lag_1']
X = df[features]

#Prolema 1: Regresión
#Predecir la cantidad numérica de turistas
y_regresion = df['Cantidad_Turistas']

#Problema 2: Clasificación
#Predecir la categoría 'Temporada_Alta' (1 o 0)
y_clasificacion = df['Temporada_Alta']

#Dividimos los datos: 70% para entrenar, 30% para probar (sin mezclar)
test_size = 0.3
split_index = int(len(df) * (1 - test_size))

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_reg, y_test_reg = y_regresion.iloc[:split_index], y_regresion.iloc[split_index:]
y_train_clas, y_test_clas = y_clasificacion.iloc[:split_index], y_clasificacion.iloc[split_index:]

#Escalado de Datos región logística y red neuronal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(f"\nDatos de entrenamiento: {len(X_train)} filas")
print(f"Datos de prueba: {len(X_test)} filas")

#Implementación de Modelos
print("\n--- INICIANDO ENTRENAMIENTO DE MODELOS ---")

#Modelo 1: regresión lineal (Regresión)
print("\n[Modelo 1: Regresión Lineal]")
modelo_reg_lineal = LinearRegression()
modelo_reg_lineal.fit(X_train, y_train_reg)
pred_reg_lineal = modelo_reg_lineal.predict(X_test)
r2_lineal = r2_score(y_test_reg, pred_reg_lineal)
print(f"R^2 (Regresión Lineal): {r2_lineal:.4f}")

#Gráfico (Real vs Predicción)
plt.figure(figsize=(10, 5))
plt.plot(y_test_reg.index, y_test_reg, label='Real')
plt.plot(y_test_reg.index, pred_reg_lineal, label='Predicción (Lineal)', linestyle='--')
plt.title('Regresión Lineal: Real vs. Predicción')
plt.legend()
plt.show()

#Modelo 2: Regresion logistica (Problema de Clasificación)
print("\n[Modelo 2: Regresión Logística]")
modelo_reg_log = LogisticRegression(random_state=42)
modelo_reg_log.fit(X_train_scaled, y_train_clas)
pred_reg_log = modelo_reg_log.predict(X_test_scaled)
acc_log = accuracy_score(y_test_clas, pred_reg_log)
print(f"Accuracy (Reg. Logística): {acc_log:.4f}")

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
print(f"Accuracy (Random Forest): {acc_rf:.4f}")

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
print(f"R^2 (Red Neuronal): {r2_nn:.4f}")

#Gráfico (Comparando todos los modelos de regresión)
plt.figure(figsize=(10, 5))
plt.plot(y_test_reg.index, y_test_reg, label='Real', color='black', lw=2)
plt.plot(y_test_reg.index, pred_reg_lineal, label=f'Predicción (Lineal) R2={r2_lineal:.2f}', linestyle='--')
plt.plot(y_test_reg.index, pred_nn, label=f'Predicción (Red Neuronal) R2={r2_nn:.2f}', linestyle=':')
plt.title('Comparación Modelos de Regresión')
plt.legend()
plt.show()

#Comparacion y resultados
print("\n--- RESUMEN DE EVALUACIÓN ---")
print("\nProblema de Regresión (Predecir Cantidad):")
print(f"  R^2 Regresión Lineal: {r2_lineal:.4f}")
print(f"  R^2 Red Neuronal    : {r2_nn:.4f}")

print("\nProblema de Clasificación (Predecir Temporada Alta):")
print(f"  Accuracy Reg. Logística: {acc_log:.4f}")
print(f"  Accuracy Random Forest : {acc_rf:.4f}")
print("\n--- FIN DEL SCRIPT ---")
