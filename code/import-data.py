
"""
Module: import_data.py
Description: Loads clinical and gene expression data, performs exploratory data visualization 
             (histograms, boxplots, count plots, and a heatmap with clustering), processes the gene expression data,
             and finally combines both datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_clinical_data(clinical_path):
    """
    Load clinical data from a CSV file.
    
    Returns:
        clinical_data (pd.DataFrame)
    """
    clinical_data = pd.read_csv(clinical_path)
    print(clinical_data.info())
    return clinical_data


def load_gene_expression_data(gene_path):
    """
    Load gene expression data from a CSV file.
    
    Returns:
        gene_data (pd.DataFrame)
    """
    gene_data = pd.read_csv(gene_path)
    print(gene_data.info())
    return gene_data


def visualize_numeric_variables(df, variables):
    """
    Visualize numeric variables with histograms and boxplots.
    """
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    
    for var in variables:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        
        # Histogram with density plot
        sns.histplot(data=df, x=var, ax=axes[0], kde=True, color='skyblue')
        axes[0].set_title(f"Histogram and Density Plot for {var}", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Frequency", fontsize=12)
        
        # Boxplot
        sns.boxplot(data=df, x=var, ax=axes[1], color='skyblue')
        axes[1].set_title(f"Boxplot for {var}", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("")
        
        plt.tight_layout()
        plt.savefig(f'analysis_{var}.png', dpi=300, bbox_inches='tight')
        plt.show()


def visualize_categorical_variables(df, variables):
    """
    Visualize categorical variables using count plots.
    """
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    
    for i, var in enumerate(variables):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        sns.countplot(x=var, data=df, ax=ax, color='skyblue', width=0.4)
        ax.set_title(f"Distribution of {var}", fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig('categorical_variables.png', dpi=300, bbox_inches='tight')
    plt.show()


def process_gene_expression_data(gene_data):
    """
    Process gene expression data:
      - Drop the first column.
      - Transpose the DataFrame.
      - Adjust column names and convert to numeric.
      - Save the processed data.
      
    Returns:
        processed_gene_data (pd.DataFrame)
    """
    # Drop the first column (assumed to be an index column)
    gene_data = gene_data.drop(gene_data.columns[0], axis=1)
    
    # Transpose the data
    gene_data = gene_data.transpose()
    
    # Set the first row as header and drop it from the data
    col_names = gene_data.iloc[0].tolist()
    gene_data.columns = col_names
    gene_data = gene_data.drop(gene_data.index[0])
    
    # Insert the index as a column 'ID_REF'
    gene_data.insert(loc=0, column='ID_REF', value=gene_data.index)
    gene_data = gene_data.reset_index(drop=True)
    
    # Convert remaining columns to numeric
    feature_columns = gene_data.columns.tolist()[1:]
    gene_data[feature_columns] = gene_data[feature_columns].astype(float)
    
    # Save the processed gene expression data
    gene_data.to_csv("gene_expression_data_processed.csv", index=False, encoding='utf-8')
    
    return gene_data


def visualize_heatmap(gene_data, top_n=5, output_file='heatmap.png'):
    """
    Visualize a clustered heatmap of the top variable genes.
    """
    # Set 'ID_REF' as index
    gene_data_indexed = gene_data.set_index('ID_REF')
    
    # Calculate variance for each gene and select top_n genes
    variances = gene_data_indexed.var().sort_values(ascending=False)
    top_genes = variances.index[:top_n]
    data_top = gene_data_indexed[top_genes]
    
    sns.clustermap(data_top, cmap="skyblue", figsize=(12, 10), row_cluster=True,
                   col_cluster=True, standard_scale=1)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def combine_datasets(clinical_data, gene_data):
    """
    Combine clinical data with gene expression data using the 'ID_REF' column.
    
    Returns:
        df_final (pd.DataFrame)
    """
    gene_data_indexed = gene_data.set_index('ID_REF')
    df_final = clinical_data.join(gene_data_indexed, on='ID_REF', how='left')
    print(df_final.info())
    df_final.to_csv("df_final.csv", index=False, encoding='utf-8')
    return df_final


def main():
    # Define file paths 
    clinical_path = "/colorectal-cancer-data.csv"
    gene_path = "/gene-expression-data.csv"
    
    # Load datasets
    clinical_df = load_clinical_data(clinical_path)
    gene_df = load_gene_expression_data(gene_path)
    
    # Visualize numeric clinical variables
    numeric_vars = ['Age (in years)', 'DFS (in months)']
    visualize_numeric_variables(clinical_df, numeric_vars)
    
    # Visualize categorical clinical variables
    categorical_vars = ['Dukes Stage', 'Gender', 'Location', 'DFS event', 'Adj_Radio', 'Adj_Chem']
    visualize_categorical_variables(clinical_df, categorical_vars)
    
    # Process gene expression data and create a heatmap
    processed_gene_df = process_gene_expression_data(gene_df)
    visualize_heatmap(processed_gene_df, top_n=5, output_file='heatmap.png')
    
    # Combine clinical and gene expression data
    combine_datasets(clinical_df, processed_gene_df)


if __name__ == '__main__':
    main()

 
  


