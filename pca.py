# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:49:05 2025

@author: HP
"""

#Import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA

# Import final DataFrame

df_path = "C:/Users/HP/Documents/Master UNIR/TFM/datbase/df_final.csv"
df = pd.read_csv(df_path)

print(df.info())


########################## DIMENSION REDUCTION ############################
###############################################################################

# 1. Dimension reduction using PCA

gene_expression_path = "C:/Users/HP/Documents/Master UNIR/TFM/database/gene_expression_data_t.csv"
gene_expression_data = pd.read_csv(gene_expression_path)

df_nuevo = gene_expression_data.set_index('ID_REF')

# 1) PCA to all the data
pca = PCA()
pca.fit(df_nuevo)

# 2) Obtain the explained varianCe for each PC 
explained_variance_ratio = pca.explained_variance_ratio_  

# 3) Create DataFrame with every PC and its explained variance  
num_components = len(explained_variance_ratio)
pc_names = [f"PC{i}" for i in range(1, num_components+1)]
df_expl = pd.DataFrame({
    "Componente Principal": pc_names,
    "Varianza Explicada (%)": explained_variance_ratio * 100,
})
df_expl["Varianza Acumulada (%)"] = df_expl["Varianza Explicada (%)"].cumsum()

# 4) Show table 
print(df_expl)

df_expl.to_csv("pca.csv", index=False, sep= ',')


# 2.  Plots 

# Obtain values 
varianza_explicada = df_expl["Varianza Explicada (%)"]
varianza_acumulada = df_expl["Varianza Acumulada (%)"]

# Explained variance plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(varianza_explicada) + 1), varianza_explicada, marker='o', linestyle='-',color= 'skyblue' )
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Explicada (%)")
plt.title("Varianza Explicada por Cada Componente Principal")
plt.grid()
plt.savefig(f'varianza_explicada.png', dpi=300, bbox_inches='tight')   

plt.show()

# Cumulative variance plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='-', color='skyblue')
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Acumulada (%)")
plt.title("Varianza Acumulada por Número de Componentes")
plt.axhline(y=95, color='r', linestyle='--', label="95% de Varianza")
plt.legend()
plt.grid()
plt.savefig(f'varianza_acumulada.png', dpi=300, bbox_inches='tight')   

plt.show()



#%%

#Selection of  Number of PC and division of the data in tranning and proof


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df_path = "C:/Users/HP/Documents/Master UNIR/TFM/database/df_final.csv"
df = pd.read_csv(df_path)
df= df.dropna()


print(df.info())

# 1. Combinar datos clínicos y datos de expresión génica
# Se utiliza la columna 'ID_REF' del DataFrame clínico y el índice del DataFrame de expresión génica para hacer la unión.

df_final = clinical_data.join(other=df_gene.set_index('ID_REF'), on='ID_REF', how='left')
print(df_final.info())

# 2. Seleccionar las características clínicas de interés
clinical_features = ['Age (in years)', 'Dukes Stage', 'Gender', 'Location', 'Adj_Radio', 'Adj_Chem']

# 3. Obtener las características de expresión génica
# Se eliminan las columnas que corresponden a datos clínicos o información que no es parte de la expresión génica.
cols_to_drop = clinical_features + ['Unnamed: 0', 'ID_REF', 'DFS (in months)', 'DFS event']
gene_expression_features = df.drop(columns=cols_to_drop)



# 4. Aplicar PCA para reducir la dimensionalidad a 20 componentes
pca = PCA(n_components=20, random_state=42)
gene_expression_pca = pca.fit_transform(gene_expression_features)


gene_expression_pca_df = pd.DataFrame(
    gene_expression_pca, 
    columns=[f'PC{i+1}' for i in range(20)],
    index=df.index
)

# 5. Construir el conjunto final de datos
# Se unen las características clínicas, la variable objetivo ('DFS event') y los componentes de PCA.

final_data = pd.concat(
    [df[clinical_features + ['DFS event']], gene_expression_pca_df],
    axis=1
)

categorical_columns = ['Gender', 'Location', 'Dukes Stage']

# Aplicamos One-Hot Encoding a las columnas categóricas

final_data_encoded = pd.get_dummies(final_data, columns=categorical_columns, drop_first=True)



# 6. Definir las variables predictoras (X) y la variable objetivo (y)
# Aquí queremos predecir el 'DFS event', es decir, si el paciente sufrió o no un evento (por ejemplo, recaída).
X_reduced = final_data_encoded.drop(columns=['DFS event'])
y_reduced = final_data_encoded['DFS event']

# 8. Dividir el conjunto de datos en entrenamiento y prueba
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced, 
    y_reduced, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_reduced  # Esto garantiza que la proporción de clases se mantenga en ambos conjuntos
)


#%%
## Implementacion de MODELS using reducted dimension data : 


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# ============================================================
# MODEL 1: Logistic Regression 
# ============================================================

# Inicializamos y entrenamos el modelo de Regresión Logística
lr_model_red = LogisticRegression(random_state=42, max_iter=1000)
lr_model_red.fit(X_train_red, y_train_red)

# Realizamos las predicciones en el conjunto de prueba
y_pred_lr_red = lr_model_red.predict(X_test_red)
# Obtenemos las probabilidades para la clase positiva (usadas en la curva ROC)
y_pred_proba_lr_red = lr_model_red.predict_proba(X_test_red)[:, 1]

# Calculamos las métricas de desempeño para Regresión Logística
accuracy_lr_red = accuracy_score(y_test_red, y_pred_lr_red)
precision_lr_red = precision_score(y_test_red, y_pred_lr_red)
recall_lr_red = recall_score(y_test_red, y_pred_lr_red)
f1_lr_red = f1_score(y_test_red, y_pred_lr_red)
roc_auc_lr_red = roc_auc_score(y_test_red, y_pred_proba_lr_red)

# Obtención de la curva ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test_red, y_pred_proba_lr_red)


# ============================================================
# MODEL 2: SVM (Support Vector Machine)
# ============================================================

# Inicializamos y entrenamos el modelo SVM (con probability=True para obtener probabilidades)
svm_model_red = SVC(probability=True, random_state=42)
svm_model_red.fit(X_train_red, y_train_red)

# Realizamos las predicciones y obtenemos las probabilidades
y_pred_svm_red = svm_model_red.predict(X_test_red)
y_pred_proba_svm_red = svm_model_red.predict_proba(X_test_red)[:, 1]

# Calculamos las métricas de desempeño para SVM
accuracy_svm_red = accuracy_score(y_test_red, y_pred_svm_red)
precision_svm_red = precision_score(y_test_red, y_pred_svm_red)
recall_svm_red = recall_score(y_test_red, y_pred_svm_red)
f1_svm_red = f1_score(y_test_red, y_pred_svm_red)
roc_auc_svm_red = roc_auc_score(y_test_red, y_pred_proba_svm_red)

# Obtención de la curva ROC
fpr_svm, tpr_svm, _ = roc_curve(y_test_red, y_pred_proba_svm_red)


# ============================================================
# MODEL 3: Random Forest
# ============================================================

# Inicializamos y entrenamos el modelo Random Forest
rf_model_red = RandomForestClassifier(random_state=42)
rf_model_red.fit(X_train_red, y_train_red)

# Realizamos las predicciones y obtenemos las probabilidades
y_pred_rf_red = rf_model_red.predict(X_test_red)
y_pred_proba_rf_red = rf_model_red.predict_proba(X_test_red)[:, 1]

# Calculamos las métricas de desempeño para Random Forest
accuracy_rf_red = accuracy_score(y_test_red, y_pred_rf_red)
precision_rf_red = precision_score(y_test_red, y_pred_rf_red)
recall_rf_red = recall_score(y_test_red, y_pred_rf_red)
f1_rf_red = f1_score(y_test_red, y_pred_rf_red)
roc_auc_rf_red = roc_auc_score(y_test_red, y_pred_proba_rf_red)

# Obtención de la curva ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test_red, y_pred_proba_rf_red)

# ============================================================
# RESULTS 
# ============================================================

# Organized data in a dictionary 
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

# Print results 
print("Resultados de la predicción del 'DFS event':")
for model_name, metrics in results_reduced.items():
    print(f"\nModelo: {model_name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")



# Curves ROC 
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Regresión logística (AUC = {roc_auc_lr_red:.2f})', color='blue')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm_red:.2f})', color='red')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf_red:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('Tasa Falso Positivo Rate')
plt.ylabel('Tasa Verdadero Positivo')
plt.title('Curvas ROC de los Modelos')
plt.legend(loc='lower right')
plt.savefig(f'CurvasRoc2.png', dpi=300, bbox_inches='tight')   
plt.show()


#%%
# Ultimo paso

import numpy as np
import pandas as pd

def top_genes_for_target(final_data, pca, gene_expression_features, target_variable_name, 
                         pca_prefix='PC', num_pcs=10, top_n=20):
    """
    Calcula las correlaciones entre cada componente principal (PC) y una variable objetivo,
    selecciona la PC con mayor correlación absoluta y extrae los genes (loadings) más relevantes.
    
    Parámetros:
      - final_data: DataFrame que contiene las columnas de las PCs y la variable objetivo.
      - pca: Objeto PCA ya ajustado (por ejemplo, el que usaste para transformar los datos génicos).
      - gene_expression_features: DataFrame o lista de nombres de los genes originales.
      - target_variable_name: Nombre de la columna de la variable objetivo (por ejemplo, 'Adj_Chem' o 'DFS event').
      - pca_prefix: Prefijo usado en el nombre de las columnas de las PCs (default: 'PC').
      - num_pcs: Número de componentes principales usadas (default: 10).
      - top_n: Número de genes (loadings) a mostrar (default: 20).
    
    Retorna:
      - sorted_correlations: Lista de tuplas (PC, correlación) ordenadas por el valor absoluto de la correlación (descendente).
      - most_correlated_pc: Nombre de la PC con mayor correlación absoluta.
      - top_genes: Serie de los top_n genes con sus cargas (loadings) ordenadas por su valor absoluto.
    """
    # Lista de nombres de las PCs a evaluar
    pca_columns = [f'{pca_prefix}{i+1}' for i in range(num_pcs)]
    
    # Extraer la variable objetivo
    target_variable = final_data[target_variable_name]
    
    # Calcular las correlaciones entre cada PC y la variable objetivo
    correlations = {
        pc: np.corrcoef(final_data[pc], target_variable)[0, 1] 
        for pc in pca_columns
    }
    
    # Ordenar las PCs por el valor absoluto de la correlación (de mayor a menor)
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Seleccionar la PC con mayor correlación absoluta
    most_correlated_pc = sorted_correlations[0][0]
    
    # Obtener el índice numérico de la PC (por ejemplo, 'PC3' -> 3, restamos 1 para indexar en pca.components_)
    pc_index = int(most_correlated_pc.replace(pca_prefix, '')) - 1
    
    # Extraer las cargas (loadings) de la PC seleccionada; 
    # se asume que gene_expression_features es un DataFrame o tiene atributo .columns con los nombres de los genes.
    loadings = pd.Series(pca.components_[pc_index], index=gene_expression_features.columns)
    
    # Ordenar los genes según el valor absoluto de su loading y seleccionar los top_n
    top_genes = loadings.reindex(loadings.abs().sort_values(ascending=False).index).head(top_n)
    
    return sorted_correlations, most_correlated_pc, top_genes

# --------------------------------------------------------------------
# Aplicación para la variable de resistencia (por ejemplo, 'Adj_Chem')
# --------------------------------------------------------------------
sorted_corr_resistance, best_pc_resistance, top_genes_resistance = top_genes_for_target(
    final_data=final_data, 
    pca=pca, 
    gene_expression_features=gene_expression_features, 
    target_variable_name='Adj_Chem',   # Variable proxy de resistencia al tratamiento
    pca_prefix='PC',
    num_pcs=10,   # Cambia a 20 si usaste 20 componentes
    top_n=20
)

print("Correlaciones entre PCs y resistencia (Adj_Chem):")
print(sorted_corr_resistance)
print(f"\nLa PC con mayor correlación absoluta con resistencia es: {best_pc_resistance}")
print("\nGenes más relevantes para resistencia al tratamiento (top 20):")
print(top_genes_resistance)

# --------------------------------------------------------------------
# Aplicación para la variable de mortalidad/progresión (por DFS event)
# --------------------------------------------------------------------
sorted_corr_mortality, best_pc_mortality, top_genes_mortality = top_genes_for_target(
    final_data=final_data, 
    pca=pca, 
    gene_expression_features=gene_expression_features, 
    target_variable_name='DFS event',  # Variable que sirve como proxy para mortalidad o progresión
    pca_prefix='PC',
    num_pcs=10,   # Cambia a 20 si es el caso
    top_n=20
)

print("\nCorrelaciones entre PCs y mortalidad (DFS event):")
print(sorted_corr_mortality)
print(f"\nLa PC con mayor correlación absoluta con mortalidad es: {best_pc_mortality}")
print("\nGenes más relevantes para mortalidad (top 20):")
print(top_genes_mortality)



