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
