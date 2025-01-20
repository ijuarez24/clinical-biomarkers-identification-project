# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:34:22 2025

@author: Inma Juárez Gonzálvez 
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generar datos sintéticos
np.random.seed(42)
n_samples = 1000

# Características
edad = np.random.normal(60, 10, n_samples)  # Edad promedio: 60 años
cea = np.random.normal(5, 2, n_samples)  # Nivel de CEA promedio: 5 ng/mL
mutacion_kras = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # KRAS: 30% mutado
mutacion_braf = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # BRAF: 10% mutado

# Etiqueta objetivo (pronóstico)
prognosis = (0.2 * edad - 0.3 * cea + 0.4 * mutacion_kras + 0.6 * mutacion_braf + np.random.normal(0, 1, n_samples)) > 10
prognosis = prognosis.astype(int)

# Crear DataFrame
data = pd.DataFrame({
    'edad': edad,
    'cea': cea,
    'mutacion_kras': mutacion_kras,
    'mutacion_braf': mutacion_braf,
    'prognosis': prognosis
})

# Dividir el conjunto de datos en test y entrenamiento
X = data[['edad', 'cea', 'mutacion_kras', 'mutacion_braf']]
y = data['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


### MODELOS ML 

# Regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Métricas modelo Regresión Logistica:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}\n")


#SVM

from sklearn.svm import SVC

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)


print(f"Métricas modelo SVM:")
print(f"Accuracy: {accuracy_svm:.2f}")
print(f"Precision: {precision_svm:.2f}")
print(f"Recall: {recall_svm:.2f}")
print(f"F1 Score: {f1_svm:.2f}\n")



#Random forest 

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)



print(f"Métricas modelo Random Forest:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1 Score: {f1_rf:.2f}")



#Graficos y cambio de hiperparametros 
