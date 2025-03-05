
"""
Module: pca_variance.py
Description: Performs PCA on gene expression data and plots the explained and cumulative variance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_pca(gene_expression_path, output_csv='pca.csv'):
    """
    Performs PCA on gene expression data and saves the explained variance per component.

    Parameters:
        gene_expression_path (str): File path to the gene expression CSV.
        output_csv (str): Output CSV file for saving the PCA table.

    Returns:
        pca (PCA): Fitted PCA object.
        df_expl (pd.DataFrame): DataFrame with explained and cumulative variance.
    """
    df_gene = pd.read_csv(gene_expression_path)
    df_gene_indexed = df_gene.set_index('ID_REF')
    
    pca = PCA()
    pca.fit(df_gene_indexed)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    num_components = len(explained_variance_ratio)
    pc_names = [f"PC{i}" for i in range(1, num_components + 1)]
    
    df_expl = pd.DataFrame({
        "Principal Component": pc_names,
        "Explained Variance (%)": explained_variance_ratio * 100,
    })
    df_expl["Cumulative Variance (%)"] = df_expl["Explained Variance (%)"].cumsum()
    
    print(df_expl)
    df_expl.to_csv(output_csv, index=False)
    return pca, df_expl

def plot_variance(df_expl):
    """
    Plots the explained and cumulative variance from PCA.

    Parameters:
        df_expl (pd.DataFrame): DataFrame with variance metrics.
    """
    var_explained = df_expl["Explained Variance (%)"]
    var_cumulative = df_expl["Cumulative Variance (%)"]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(var_explained) + 1), var_explained, marker='o', linestyle='-', color='skyblue')
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance (%)")
    plt.title("Explained Variance by Each Principal Component")
    plt.grid()
    plt.savefig('explained_variance.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(var_cumulative) + 1), var_cumulative, marker='o', linestyle='-', color='skyblue')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance (%)")
    plt.title("Cumulative Variance by Number of Components")
    plt.axhline(y=95, color='r', linestyle='--', label="95% Variance")
    plt.legend()
    plt.grid()
    plt.savefig('cumulative_variance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Define the path to the gene expression CSV file (update as needed)
    gene_expression_path = "C:/Users/HP/Documents/Master UNIR/TFM/database/gene_expression_data_t.csv"
    
    # Perform PCA and plot variance metrics
    pca, df_expl = perform_pca(gene_expression_path)
    plot_variance(df_expl)

if __name__ == '__main__':
    main()
