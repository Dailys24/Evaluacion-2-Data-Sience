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
