
"""
Module: pca_ml.py
Description: Applies PCA on combined clinical and gene expression data and implements ML models on the reduced dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

def ml_modeling_with_pca(final_data, gene_expression_features, n_components=20):
    """
    Applies PCA on gene expression features, combines with clinical data,
    performs one-hot encoding on categorical variables, splits the data,
    trains ML models, and plots ROC curves.

    Parameters:
        final_data (pd.DataFrame): Combined dataset with clinical and gene expression data.
        gene_expression_features (list or pd.Index): List of gene expression feature columns.
        n_components (int): Number of principal components to retain.

    Returns:
        final_data_encoded (pd.DataFrame): DataFrame after one-hot encoding.
        pca (PCA): Fitted PCA object.
    """
    # Apply PCA on gene expression features
    pca = PCA(n_components=n_components, random_state=42)
    gene_expression_pca = pca.fit_transform(final_data[gene_expression_features])
    
    gene_expression_pca_df = pd.DataFrame(
        gene_expression_pca, 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=final_data.index
    )
    
    # Define clinical features (update as needed)
    clinical_features = ['Age (in years)', 'Dukes Stage', 'Gender', 'Location', 'Adj_Radio', 'Adj_Chem', 'DFS (in months)', 'DFS event']
    
    # Combine clinical data with PCA components
    combined_data = pd.concat([final_data[clinical_features], gene_expression_pca_df], axis=1)
    
    # One-Hot Encoding for categorical variables
    categorical_columns = ['Gender', 'Location', 'Dukes Stage']
    final_data_encoded = pd.get_dummies(combined_data, columns=categorical_columns, drop_first=True)
    
    # Define predictors and target
    X = final_data_encoded.drop(columns=['DFS event'])
    y = final_data_encoded['DFS event']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train ML models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n{name} Model Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        
        results[name] = y_proba
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, y_proba in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        color = {'Logistic Regression': 'blue', 'SVM': 'red', 'Random Forest': 'green'}.get(name)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.2f})', color=color)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Models with PCA Reduced Data')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves_pca_ml.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return final_data_encoded, pca

def main():
    # Define the path to the final combined data CSV file (update as needed)
    final_data_path = "C:/Users/HP/Documents/Master UNIR/TFM/datbase/df_final.csv"
    df_final = pd.read_csv(final_data_path)
    df_final = df_final.dropna()
    
    # Define gene expression feature columns by excluding clinical columns
    clinical_features = ['Age (in years)', 'Dukes Stage', 'Gender', 'Location', 'Adj_Radio', 'Adj_Chem', 'DFS (in months)', 'DFS event']
    gene_expression_features = df_final.columns.difference(clinical_features)
    
    # Apply PCA and ML modeling on the reduced data
    final_data_encoded, pca = ml_modeling_with_pca(df_final, gene_expression_features, n_components=20)
    
if __name__ == '__main__':
    main()
