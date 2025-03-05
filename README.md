# Master´s Thesis in Bioinformatics: Identification of clinical biomarkers in colorectal cancer

This repository contains the code and resources for my Master's Thesis project in Bioinformatics. The focus of this project is the identification of clinical markers in colorectal cancer. 

## Table of Contents
- [General Information](#general-information)
- [Project Structure](#project-structure)
- [Synthetic Data Base](#synthetic-data-base)
- [Real Data Base](#real-data-base)
- [Current Status](#current-status)
- [Contact](#contact)
- [Genetic Analysis](#genetic-analysis)



## General Information
Colorectal cancer is one of the most prevalent oncological diseases worldwide. Bioinformatics, combined with artificial intelligence and machine learning, offers new opportunities to improve its detection and prognosis. This project explores the use of these tools to advance personalized medicine and optimize patient diagnosis and treatment.

Key aspects of this project include:
- Identification of potential clinical markers in colorectal cancer using bioinformatics tools.
- Implementation of machine learning models to evaluate and predict the prognosis of the disease.

## Project Structure 
```
clinical-biomarkers-identification-project/
│
├── data/
│ ├── gene-expression-data.csv
│ ├── colorectal-cancer-data.csv
│ ├── df-final.csv
├── code/
│ ├── synthetic-data.py
│ ├── import-data.py
│ ├── pca-variance.py
│ ├── pca-ml.py
│ ├── genetic-analysis.py
└── README.md
````

## Synthetic Data Base

This project applies supervised learning models to predict the prognosis of colorectal cancer based on clinical data, specifically focusing on genetic mutations associated with the disease.

### Overview
A synthetic dataset was created for this study, including patient demographics and key clinical biomarkers. The dataset simulates real-world scenarios to evaluate different machine learning models. The biomarkers considered include:
- CEA levels (protein biomarker)
- KRAS and BRAF mutations (genetic biomarkers)

The dataset assumes:
- 30% of patients have KRAS mutations
- 10% of patients have BRAF mutations, which are more aggressive

### Methodology
The prognosis is determined by combining these factors using assigned weights:
- Age positively contributes to prognosis
- Higher CEA levels worsen prognosis
- Genetic mutations have different impacts, with BRAF being more significant than KRAS

### Models & Evaluation
The following machine learning models were implemented:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
  
Performance metrics used: Accuracy, F1-score, AUC, Recall, and Precision. Hyperparameter tuning was also explored to improve predictive accuracy.

### Predictions
Once trained, the models can be used to assess new patients:

- Probability < 0.5 → Good prognosis
- Probability > 0.5 → Poor prognosis
- Probability ≈ 0.5 → Uncertain prognosis

### Conclusion
While the dataset is synthetic, this approach can be applied in real medical settings. With improved data quality, hospitals could develop databases for personalized medicine, enabling genetic screening for patients at risk and optimizing treatment strategies.

This project demonstrates the potential of machine learning in advancing colorectal cancer prognosis and personalized healthcare.


## Real Data Base 

### Overview

This chapter focuses on analyzing publicly available datasets related to colorectal cancer. The objective is to preprocess and visualize the data, implement machine learning (ML) models, and extract insights about genetic factors associated with treatment resistance, tumor recurrence, and patient survival.

### Dataset Description

Two datasets were used:

- Colorectal Cancer Clinical Data: Contains information from 62 patients who underwent tumor resection, including age, Dukes staging, tumor location, treatment details, and disease-free survival (DFS).

- Gene Expression Data: Includes gene expression levels for approximately 2000 genes in the same patients, preprocessed using a log₂ transformation.

### Data Preprocessing and Visualization

The first step involved visualizing clinical data distributions through histograms, density plots, and boxplots. Key insights include:

- Most patients are in intermediate Dukes stages (B or C).

- The average age is 61 years, with DFS varying widely from 4 to 108 months.

- Treatment distribution for chemotherapy and radiotherapy was balanced.

For gene expression data, a heatmap with hierarchical clustering was used to explore gene variation across patients. Principal Component Analysis (PCA) was applied to reduce dimensionality, retaining 20 components that explain 71% of the variance.

### Machine Learning Implementation

ML models were trained to predict DFS event (whether a patient experienced disease recurrence or death). The steps included:

- Merging clinical and gene expression data.

- Encoding categorical variables.

- Splitting data into training (80%) and testing (20%) sets.

- Implementing supervised learning models:

      - Logistic Regression (interpretable and widely used in medicine).

      - Support Vector Machine (SVM) (captures complex relationships).

      - Random Forest (robust to noise and handles high-dimensional data).


### Model Performance Evaluation

Models were compared based on accuracy, precision, recall, F1-score, and AUC (Area Under the Curve). Key findings:

- Random Forest had the highest overall accuracy and F1-score.

- Logistic Regression was the most interpretable, making it valuable for clinical decision-making.

- SVM demonstrated strong discrimination capabilities but required more computational resources.

### Conclusion and Future Steps

This study demonstrates the potential of ML in colorectal cancer prognosis by integrating clinical and genomic data. Future work could involve:

- Refining models with hyperparameter tuning.

- Expanding datasets to improve generalizability.

- Applying deep learning techniques for enhanced predictive accuracy.

By leveraging bioinformatics and AI, this approach contributes to precision medicine, supporting personalized treatment strategies for colorectal cancer patients.


## Genetic Analysis



## Current Status 
This project is currently completed. 

## Contact
- [LinkedIn Profile](https://www.linkedin.com/in/inmaculadajuarez)
- [Email](mailto:inma.juarez24@gmail.com)  

