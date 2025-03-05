
"""
Module: genetic_analysis.py
Description: Performs genetic analysis by correlating PCA components with target variables 
             (e.g., treatment resistance and mortality/progression) to identify top genes associated with these outcomes.
"""

import pandas as pd
import numpy as np

def top_genes_for_target(final_data, pca, gene_expression_features, target_variable_name, 
                         pca_prefix='PC', num_pcs=10, top_n=20):
    """
    Computes correlations between each principal component (PC) and a target variable,
    selects the PC with the highest absolute correlation, and extracts the top genes (loadings).

    Parameters:
      - final_data (pd.DataFrame): DataFrame containing the PC columns and the target variable.
      - pca (PCA): Fitted PCA object.
      - gene_expression_features (pd.DataFrame): DataFrame of original gene expression features.
      - target_variable_name (str): Name of the target variable column.
      - pca_prefix (str): Prefix for PC column names (default: 'PC').
      - num_pcs (int): Number of principal components used (default: 10).
      - top_n (int): Number of top genes (loadings) to return (default: 20).

    Returns:
      - sorted_correlations (list): List of tuples (PC, correlation) sorted by absolute correlation.
      - most_correlated_pc (str): The PC with the highest absolute correlation.
      - top_genes (pd.Series): Series of the top genes with their loadings.
    """
    pca_columns = [f'{pca_prefix}{i+1}' for i in range(num_pcs)]
    
    target_variable = final_data[target_variable_name]
    correlations = {pc: np.corrcoef(final_data[pc], target_variable)[0, 1] for pc in pca_columns}
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    most_correlated_pc = sorted_correlations[0][0]
    pc_index = int(most_correlated_pc.replace(pca_prefix, '')) - 1
    
    # Extract loadings for the most correlated PC
    loadings = pd.Series(pca.components_[pc_index], index=gene_expression_features.columns)
    top_genes = loadings.reindex(loadings.abs().sort_values(ascending=False).index).head(top_n)
    
    return sorted_correlations, most_correlated_pc, top_genes

def run_genetic_analysis(final_data, pca, gene_expression_features):
    """
    Runs the genetic analysis to identify genes most associated with treatment resistance and mortality/progression.

    Parameters:
        final_data (pd.DataFrame): DataFrame containing PCA components and target variables.
        pca (PCA): Fitted PCA object.
        gene_expression_features (pd.DataFrame): DataFrame of original gene expression features.
    """
    # Analysis for treatment resistance (Adj_Chem)
    sorted_corr_res, best_pc_res, top_genes_res = top_genes_for_target(
        final_data=final_data, 
        pca=pca, 
        gene_expression_features=gene_expression_features, 
        target_variable_name='Adj_Chem',  
        pca_prefix='PC',
        num_pcs=10,   
        top_n=20
    )
    
    print("Correlations between PCs and treatment resistance (Adj_Chem):")
    print(sorted_corr_res)
    print(f"\nThe PC with the highest absolute correlation with resistance is: {best_pc_res}")
    print("\nTop 20 genes relevant for treatment resistance:")
    print(top_genes_res)
    
    # Analysis for mortality/progression (DFS event)
    sorted_corr_mort, best_pc_mort, top_genes_mort = top_genes_for_target(
        final_data=final_data, 
        pca=pca, 
        gene_expression_features=gene_expression_features, 
        target_variable_name='DFS event',  
        pca_prefix='PC',
        num_pcs=10,   
        top_n=20
    )
    
    print("\nCorrelations between PCs and mortality/progression (DFS event):")
    print(sorted_corr_mort)
    print(f"\nThe PC with the highest absolute correlation with mortality is: {best_pc_mort}")
    print("\nTop 20 genes relevant for mortality/progression:")
    print(top_genes_mort)

def main():
    # Define the path to the final combined data CSV file (update as needed)
    final_data_path = "/df_final.csv"
    df_final = pd.read_csv(final_data_path)
    df_final = df_final.dropna()
    
    # Define clinical columns and extract gene expression features
    clinical_features = ['Age (in years)', 'Dukes Stage', 'Gender', 'Location', 'Adj_Radio', 'Adj_Chem', 'DFS (in months)', 'DFS event']
    gene_expression_features = df_final.drop(columns=clinical_features)
    
    # For this analysis, we need PCA components in the final data.
    # If not present, we recompute PCA on gene expression features with 10 components.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10, random_state=42)
    pca_components = pca.fit_transform(gene_expression_features)
    for i in range(10):
        df_final[f'PC{i+1}'] = pca_components[:, i]
    
    run_genetic_analysis(df_final, pca, gene_expression_features)

if __name__ == '__main__':
    main()
