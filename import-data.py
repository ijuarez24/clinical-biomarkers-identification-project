# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:14:54 2025

@author: HP
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORTAR DATOS 

#Cargar base de datos

clinical_path = "C:/Users/HP/Documents/Master UNIR/TFM/colorectal-cancer-data.csv"
clinical_data = pd.read_csv(clinical_path)
df_gene_path= "C:/Users/HP/Documents/Master UNIR/TFM/gene-expression-data.csv"
df_gene = pd.read_csv(df_gene_path)



#CLINICAL DATA 

print(clinical_data.info())


# VISUALIZACIÓN 

# Variables categoricas 

variables_num = ['Age (in years)', 'DFS (in months)']

# basic stats
clinical_data[variables_num].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])

# Configurar estilo y paleta de colores
sns.set_style('whitegrid')       # Estilo de la cuadrícula
sns.set_palette('Set2')          # Paleta de colores (puedes cambiarla por otra)

for f in variables_num:
    # Crear figura y ejes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    # Histograma con Seaborn
    sns.histplot(
        data=clinical_data, 
        x=f, 
        ax=axes[0], 
        kde=True,               # Agrega la curva de densidad
        color='skyblue'         # Color del histograma (puedes cambiarlo)
    )
    axes[0].set_title(f"Histograma y Curva de Desidad {f}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("")                 # Eliminar etiqueta de eje X
    axes[0].set_ylabel("Frecuencia", fontsize=12)
    
    # Boxplot con Seaborn (horizontal)
    sns.boxplot(
        data=clinical_data, 
        x=f, 
        ax=axes[1],
        color='skyblue'
    )
    axes[1].set_title(f"Boxplot {f}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("")
    
    # Ajustar el espacio entre gráficos
    plt.tight_layout()
    
    plt.savefig(f'analisis_{f}.png', dpi=300, bbox_inches='tight')   
 
    # Mostrar figura
    plt.show()




# Variables numericas 

sns.set_style("whitegrid")
sns.set_palette("Set2")

variable_num = ['Dukes Stage', 'Gender', 'Location', 
                'DFS event', 'Adj_Radio', 'Adj_Chem']

# Crear la figura y la cuadrícula de subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))



for i, cat in enumerate(variable_num):
    # Determinar la posición en la cuadrícula
    row = i // 3
    col = i % 3
    
    ax = axes[row, col]
    
    # Crear un countplot con un color específico y barras más estrechas
    sns.countplot(
        x=cat, 
        data=clinical_data, 
        ax=ax, 
        color='skyblue',   # <--- Aquí forzamos el color de las barras
        width=0.4
    )
    
    ax.set_title(f"Distribución de {cat}", fontsize=12, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Frecuencia")
    
    # Girar etiquetas del eje X si son largas
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

# Ajustar el espacio entre subplots
plt.tight_layout()

#Guardar imagen

plt.savefig(f'variable_numerica.png', dpi=300, bbox_inches='tight')   


# Mostrar la figura con todos los subplots en una imagen
plt.show()



# GENE EXPRESSION

#Procesado 

print(df_gene.info())

df_gene = df_gene.drop(df_gene.columns[0], axis = 1) # eliminar primera columna (numeros)

df_gene.head()

# Trasponer datos

df_gene = df_gene.transpose()


# EXPLICACION DE ESTA PARTE 

col_names = df_gene.iloc[0].tolist()
df_gene.columns = col_names
# now remove redundant first row
df_gene = df_gene.drop(axis=0, index='ID_REF')
# add ID_REF (as first column) by copying index
df_gene.insert(loc=0, column='ID_REF', value=df_gene.index)
# and reset index
df_gene = df_gene.reset_index(drop=True)
# convert to numerical
features_gene_num = df_gene.columns.tolist()[1:]
df_gene[features_gene_num] = df_gene[features_gene_num].astype(float)


#Guardar excel final ya trasnpouesto
df_gene.to_csv("gene_expression_data_t.csv", index=False, encoding='utf-8')




# Visualización (heatmap) 


# Supongamos que `expression_data` es un DataFrame de 64 filas (pacientes) y 2000 columnas (genes).
# A veces se seleccionan menos genes (e.g., top 100 por varianza) para que el heatmap sea más legible.


df_nuevo = df_gene.set_index('ID_REF')

# Extraer top 100 genes más variables (ejemplo)
variances = df_nuevo.var().sort_values(ascending=False)
top_genes = variances.index[:5]
data_top = df_nuevo[top_genes]

# Crear un mapa de calor con clustering
sns.clustermap(
    data_top,
    cmap="skyblue",
    figsize=(12, 10),
    row_cluster=True,     # Agrupar pacientes
    col_cluster=True,     # Agrupar genes
    standard_scale=1      # Escala cada columna (gen) para comparaciones
)

plt.savefig(f'heatmap.png', dpi=300, bbox_inches='tight')   

plt.show()

    
    

#Combinar las dos bases de datos 

df_final = clinical_data.join(other=df_gene.set_index('ID_REF'), on='ID_REF', how='left')
print(df_final.info())

df_final.to_csv("df_final.csv", index=False, encoding='utf-8')
