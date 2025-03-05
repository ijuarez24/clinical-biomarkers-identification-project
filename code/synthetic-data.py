
"""
Module: synthetic_data.py
Description: Generates synthetic clinical data, splits the dataset, trains several ML models 
             (Logistic Regression, SVM, Random Forest), prints evaluation metrics, and plots ROC curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generate a synthetic dataset with clinical features and a prognosis label.
    
    Returns:
        data (pd.DataFrame): DataFrame containing features and the prognosis label.
    """
    np.random.seed(random_state)
    # Features
    age = np.random.normal(60, 10, n_samples)         # Average age: 60 years
    cea = np.random.normal(5, 2, n_samples)             # Average CEA level: 5 ng/mL
    kras_mutation = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    braf_mutation = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Prognosis label (binary): a linear combination of features with added noise
    prognosis = (0.2 * age - 0.3 * cea + 0.4 * kras_mutation + 0.6 * braf_mutation +
                 np.random.normal(0, 1, n_samples)) > 10
    prognosis = prognosis.astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'cea': cea,
        'kras_mutation': kras_mutation,
        'braf_mutation': braf_mutation,
        'prognosis': prognosis
    })
    return data


def split_data(data, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = data[['age', 'cea', 'kras_mutation', 'braf_mutation']]
    y = data['prognosis']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression, SVM, and Random Forest models and print evaluation metrics.
    
    Returns:
        results (dict): Dictionary with predicted probabilities for each model (for ROC plotting).
    """
    results = {}
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} Model Metrics:")
        print(f"  Accuracy:  {accuracy:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1 Score:  {f1:.2f}\n")
        
        # Save predicted probabilities for ROC curve plotting
        if hasattr(model, "predict_proba"):
            results[name] = model.predict_proba(X_test)[:, 1]
        else:
            results[name] = None
    return results, y_test


def plot_roc_curves(y_test, proba_dict, output_file='roc_curves.png'):
    """
    Plot ROC curves for each model based on predicted probabilities.
    """
    plt.figure(figsize=(8, 6))
    colors = {'Logistic Regression': 'blue', 'SVM': 'red', 'Random Forest': 'green'}
    for model_name, y_proba in proba_dict.items():
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', color=colors.get(model_name))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Models')
    plt.legend(loc='lower right')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Generate synthetic data and split into train/test sets
    data = generate_synthetic_data()
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train models and display metrics
    proba_dict, y_test = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot ROC curves for each model
    plot_roc_curves(y_test, proba_dict)


if __name__ == '__main__':
    main()



