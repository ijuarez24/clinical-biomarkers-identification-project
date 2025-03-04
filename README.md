# Master´s Thesis in Bioinformatics: Identification of clinical biomarkers in colorectal cancer

This repository contains the code and resources for my Master's Thesis project in Bioinformatics. The focus of this project is the identification of clinical markers in colorectal cancer. 

## Table of Contents
- [General Information](#general-information)
- [Project Structure](#project-structure)
- [Synthetic Data Base](#synthetic-data-base)
- [Real Data Base](#real-data-base)
- [Current Status](#current-status)
- [Contact](#contact)



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
│ ├── database-sintetica.py
│ ├── import-data.py
│ ├── database-real.py
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


### Data Visualization 


### PCA


### Machine Learning Algorithms Implementation 


### Results 


## Current Status 
This project is currently completed, waiting for final evaluations.

## Contact
- [LinkedIn Profile](https://www.linkedin.com/in/inmaculadajuarez)
- [Email](mailto:inma.juarez24@gmail.com)  

