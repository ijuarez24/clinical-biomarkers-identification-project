# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:47:19 2025

@author: Inma Juárez Gonzálvez 
"""

#Combianr las dos bases de datos: Al introducir una base de datso sobre genes podemos predecir que genes influyen tanto en la resistencia al tratamiento y la supervivencia. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA

# Cargar la bases de datos gene expression 

file_path = "C:/Users/HP/Documents/Master UNIR/TFM/colorectal-cancer-data.csv"
clinical_data = pd.read_csv(file_path)
gene_expression_path = "C:/Users/HP/Documents/Master UNIR/TFM/gene-expression-data.csv"
gene_expression_data = pd.read_csv(gene_expression_path)



# Preprocesamiento de datos clínicos
clinical_data_cleaned = clinical_data.dropna().copy()
categorical_columns = ['Dukes Stage', 'Gender', 'Location']
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col, encoder in label_encoders.items():
    clinical_data_cleaned[col] = encoder.fit_transform(clinical_data_cleaned[col])
clinical_data_cleaned = clinical_data_cleaned.rename(columns={"ID_REF": "Sample_ID"})

# Transponer datos de expresión génica
gene_expression_data_transposed = gene_expression_data.set_index('ID_REF').T
gene_expression_data_transposed.index.name = "Sample_ID"

# Combinar datos
combined_data = pd.merge(clinical_data_cleaned, gene_expression_data_transposed, left_on="Sample_ID", right_index=True, how="inner")

# Separar características clínicas y génicas
clinical_features = ['Age (in years)', 'Dukes Stage', 'Gender', 'Location', 'Adj_Radio', 'Adj_Chem']
gene_expression_features = combined_data.drop(columns=clinical_features + ['Unnamed: 0', 'Sample_ID', 'DFS (in months)', 'DFS event'])

# Aplicar PCA para reducir las características génicas
pca = PCA(n_components=10, random_state=42)
gene_expression_pca = pca.fit_transform(gene_expression_features)
gene_expression_pca_df = pd.DataFrame(gene_expression_pca, columns=[f'PC{i+1}' for i in range(10)])

# Crear conjunto final con características reducidas
final_data = pd.concat([combined_data[clinical_features + ['DFS event']], gene_expression_pca_df], axis=1)

# Definir X y y
X_reduced = final_data.drop(columns=['DFS event'])
y_reduced = final_data['DFS event']

# Dividir en conjunto de entrenamiento y prueba
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced, y_reduced, test_size=0.2, random_state=42, stratify=y_reduced
)

# Modelo de Regresión Logística
lr_model_red = LogisticRegression(random_state=42, max_iter=1000)
lr_model_red.fit(X_train_red, y_train_red)
y_pred_lr_red = lr_model_red.predict(X_test_red)
y_pred_proba_lr_red = lr_model_red.predict_proba(X_test_red)[:, 1]

# Métricas para Regresión Logística
accuracy_lr_red = accuracy_score(y_test_red, y_pred_lr_red)
precision_lr_red = precision_score(y_test_red, y_pred_lr_red)
recall_lr_red = recall_score(y_test_red, y_pred_lr_red)
f1_lr_red = f1_score(y_test_red, y_pred_lr_red)
roc_auc_lr_red = roc_auc_score(y_test_red, y_pred_proba_lr_red)

# Modelo SVM
svm_model_red = SVC(probability=True, random_state=42)
svm_model_red.fit(X_train_red, y_train_red)
y_pred_svm_red = svm_model_red.predict(X_test_red)
y_pred_proba_svm_red = svm_model_red.predict_proba(X_test_red)[:, 1]

# Métricas para SVM
accuracy_svm_red = accuracy_score(y_test_red, y_pred_svm_red)
precision_svm_red = precision_score(y_test_red, y_pred_svm_red)
recall_svm_red = recall_score(y_test_red, y_pred_svm_red)
f1_svm_red = f1_score(y_test_red, y_pred_svm_red)
roc_auc_svm_red = roc_auc_score(y_test_red, y_pred_proba_svm_red)

# Modelo Random Forest
rf_model_red = RandomForestClassifier(random_state=42)
rf_model_red.fit(X_train_red, y_train_red)
y_pred_rf_red = rf_model_red.predict(X_test_red)
y_pred_proba_rf_red = rf_model_red.predict_proba(X_test_red)[:, 1]

# Métricas para Random Forest
accuracy_rf_red = accuracy_score(y_test_red, y_pred_rf_red)
precision_rf_red = precision_score(y_test_red, y_pred_rf_red)
recall_rf_red = recall_score(y_test_red, y_pred_rf_red)
f1_rf_red = f1_score(y_test_red, y_pred_rf_red)
roc_auc_rf_red = roc_auc_score(y_test_red, y_pred_proba_rf_red)

# Resultados finales
results_reduced = {
    "Logistic Regression": {
        "Accuracy": accuracy_lr_red,
        "Precision": precision_lr_red,
        "Recall": recall_lr_red,
        "F1 Score": f1_lr_red,
        "ROC AUC": roc_auc_lr_red
    },
    "SVM": {
        "Accuracy": accuracy_svm_red,
        "Precision": precision_svm_red,
        "Recall": recall_svm_red,
        "F1 Score": f1_svm_red,
        "ROC AUC": roc_auc_svm_red
    },
    "Random Forest": {
        "Accuracy": accuracy_rf_red,
        "Precision": precision_rf_red,
        "Recall": recall_rf_red,
        "F1 Score": f1_rf_red,
        "ROC AUC": roc_auc_rf_red
    }
}

print("Resultados:")
print(results_reduced)


#%%

#Comprobar componentes principales

## GENES RESPONSABLES DE LA RESISTENCIA AL TRATAMIENTO 


import numpy as np

# Seleccionar la variable de resistencia (en este caso: Adj_Chem)
resistance_variable = final_data['Adj_Chem']

# Calcular las correlaciones entre las PCs y la variable de resistencia
pca_columns = [f'PC{i+1}' for i in range(10)]
correlations = {pc: np.corrcoef(final_data[pc], resistance_variable)[0, 1] for pc in pca_columns}

# Ordenar las PCs por su correlación absoluta con la resistencia
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

# Seleccionar la PC más correlacionada
most_correlated_pc = sorted_correlations[0][0]

# Extraer las cargas (loadings) de los genes en esa PC
loadings = pd.Series(pca.components_[int(most_correlated_pc[2:]) - 1], index=gene_expression_features.columns)
top_genes = loadings.sort_values(key=abs, ascending=False).head(20)  # Los 20 genes más importantes

# Resultados: top genes responsables de la variación en la PC relacionada con resistencia
print("Correlaciones entre PCs y resistencia:", sorted_correlations)
print("Genes más relevantes para resistencia al tratamiento (top 20):")
print(top_genes)



## GENES RESPOSNABLES DE LA MORTALIDAD 

# Correlacionar las PCs con la mortalidad/progresión de la enfermedad (usando DFS event como proxy)
mortality_variable = final_data['DFS event']

# Calcular las correlaciones entre las PCs y la variable de mortalidad
correlations_mortality = {pc: np.corrcoef(final_data[pc], mortality_variable)[0, 1] for pc in pca_columns}

# Ordenar las PCs por su correlación absoluta con la mortalidad
sorted_correlations_mortality = sorted(correlations_mortality.items(), key=lambda x: abs(x[1]), reverse=True)

# Seleccionar la PC más correlacionada con mortalidad
most_correlated_pc_mortality = sorted_correlations_mortality[0][0]

# Extraer las cargas (loadings) de los genes en esa PC
loadings_mortality = pd.Series(
    pca.components_[int(most_correlated_pc_mortality[2:]) - 1], 
    index=gene_expression_features.columns
)
top_genes_mortality = loadings_mortality.sort_values(key=abs, ascending=False).head(20)  # Los 20 genes más importantes

# Resultados: top genes responsables de la variación en la PC relacionada con mortalidad
print("Correlaciones entre PCs y mortalidad:", sorted_correlations_mortality)
print("Genes más relevantes para mortalidad (top 20):")
print(top_genes_mortality)















