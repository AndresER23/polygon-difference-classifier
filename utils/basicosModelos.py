import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestRegressor

def cargar_datos(gdf_google, features, target):
    """
    Selecciona las variables predictoras y la variable objetivo.
    """
    X = gdf_google[features]
    y = gdf_google[target]
    return X, y

def escalar_datos(X_train, X_test):
    """
    Escala las variables predictoras.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def dividir_datos(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def entrenar_modelo(X_train, y_train, tipo='logistica'):
    """
    Entrena un modelo de aprendizaje supervisado basado en el tipo especificado.
    
    Parámetros:
    X_train -- Datos de entrenamiento (variables independientes)
    y_train -- Etiquetas de entrenamiento (variable dependiente)
    tipo -- El tipo de modelo a entrenar: 'logistica', 'lineal', 'random_forest'
    
    Retorna:
    El modelo entrenado
    """
    modelos = {
        'logistica': LogisticRegression(),
        'lineal': LinearRegression(),
        'random_forest': RandomForestRegressor()
    }
    
    if tipo not in modelos:
        raise ValueError("Tipo de modelo no reconocido. Use 'logistica', 'lineal' o 'random_forest'.")
    
    model = modelos[tipo]
    model.fit(X_train, y_train)
    return 

def optimizar_hiperparametros_logistica(X_train, y_train):
    """
    Optimiza hiperparámetros de regresión logística usando GridSearchCV.
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)
    print("Mejores hiperparámetros:", grid_search.best_params_)
    return grid_search.best_estimator_