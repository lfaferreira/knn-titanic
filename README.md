# KNN Classification

This project aims to utilize the K-Nearest Neighbors (KNN) algorithm for data classification, primarily focusing on identifying survivors of the Titanic shipwreck. The approach encompasses an initial phase of exploratory analysis, followed by the application of KNN, the use of validation techniques, and ultimately, hyperparameter optimization. This comprehensive approach will enable a deeper and more effective understanding of the survival prediction task in the context of the Titanic dataset.


![vist Card](https://cdn.discordapp.com/attachments/1173750582092251139/1175485464438907111/titanic.webp?ex=656b6726&is=6558f226&hm=96ea5d19025c1c1a4982cf0a8a8964f7d8fd94711f1d9009a6af9e362bc83d63&)


## Table of Contents

- [1 - Introduction](#1-introduction)

- [2 - Data Understanding](#2-data-understanding)

- [3 - Action Plan](#3-action-plan)

- [4 - Exploratory Data Analysis](#4-exploratory-data-analysis)

- [5 - Machine Learning Exploration](#5-machine-learning-exploration)

- [6 - Submissions Results](#6-submissions-results)

- [7 - Models Comparison](#7-models-comparison)

- [8 - Conclusion](#8-conclusion)
  
  

## 1 - Introduction
This project aims to conduct an **exploratory** analysis of the Titanic data set and employ the **KNN (K-Nearest Neighbors)** model to classify the survivors of the shipwreck. The widely recognized **Titanic** dataset is available at **[Kaggle](https://www.kaggle.com/c/titanic/data)**, **[Stanford](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)** and also at **[Department of Biostatistics](https://hbiostat.org/data/)**. Despite being a simple dataset, designed for studies and focused on solving a single question, it is possible to apply advanced data analysis techniques, extract insights and knowledge in a similar way to what would be done in more complex and challenging datasets.
  

## 2 - Data Understanding

The Titanic dataset, available on **Kaggle**, is divided into **train** and **test** files. The train file comprises 891 labels and 12 features, while the test file contains 418 labels and 11 features. The data is categorized into numeric and categorical types, and there are no instances of duplicate or missing values. 

The **Train** naming dictionary is as follows:

| Attribute    | Definition                          | Data Type   |
|--------------|-------------------------------------|-------------|
| PassengerId  | Passenger ID                        | int64       |
| Survived     | Survival status (0 = No, 1 = Yes)   | int64       |
| Pclass       | Ticket class                        | int64       |
| Name         | Passenger's name                    | object      |
| Sex          | Gender                              | object      |
| Age          | Age of the passenger                | float64     |
| SibSp        | Number of siblings/spouses aboard   | int64       |
| Parch        | Number of parents/children aboard   | int64       |
| Ticket       | Ticket number                       | object      |
| Fare         | Passenger fare                      | float64     |
| Cabin        | Cabin number                        | object      |
| Embarked     | Port of embarkation (C, Q, S)       | object      |

The **Test** naming dictionary is as follows:

| Attribute    | Definition                          | Data Type   |
|--------------|-------------------------------------|-------------|
| PassengerId  | Passenger ID                        | int64       |
| Pclass       | Ticket class                        | int64       |
| Name         | Passenger's name                    | object      |
| Sex          | Gender                              | object      |
| Age          | Age of the passenger                | float64     |
| SibSp        | Number of siblings/spouses aboard   | int64       |
| Parch        | Number of parents/children aboard   | int64       |
| Ticket       | Ticket number                       | object      |
| Fare         | Passenger fare                      | float64     |
| Cabin        | Cabin number                        | object      |
| Embarked     | Port of embarkation (C, Q, S)       | object      |


## 3 - Action Plan

### 3.1 - Objective

After completing the project of exploratory analysis and implementation of [KNN in the Wine database](https://github.com/lfaferreira/knn-wine), I decided to further deepen the use of this model by incorporating more sophisticated techniques. This repository aims to explore the capabilities of the KNN algorithm, focusing on optimization and enhancement to solve the Titanic problem.

### 3.2 - Tools and Frameworks

Scope of tools used in the project:

- Python 3.11.5

- Jupyter Notebook

- Git & GitHub

- Kaggle

- Pandas

- Numpy

- Scipy

- Scikit-Learn

- Machine Learning Classification Model
  - K-Nearest Neighbors
  - Gradient Boosting Classifier


  
## 4 - Exploratory Data Analysis

### 4.1 - General Train Dataset
- **Features:** 12 (including the target)
- **Labels:** 891
- **Numeric data types:** float64, int64
- **Categorical data types:** object    
- **Missing values total:** 866
- **No duplicate values**
- **Target contains 2 options:** 0, 1 (int64)


### 4.2 - Missing values
- **Age:** has 19.87% missing data that will be redefined with the median age
- **Cabin:** has 77.11% missing data, which makes it a great candidate to be removed. But this data will be grouped and transformed into numbers.
- **Embarked:** has only 0.23% missing data, which will be removed.


### 4.3 - Correlation
- The **highest correlation** in the dataset is between the features **Parch** and **Sibsp.** Both features are about family relationships, which at first glance justifies this relationship.
- The **lowest correlation** in the dataset is between the features **Pclass** and **Fare.** At first glance, it's strange to understand how passenger fare data has such a low correlation with socio-economic status.


### 4.4 - Outliers
- There are outliers in **4 features** of the dataset. I usually remove them, especially when the models used are sensitive to them. The percentage of outliers per feature is:
  - **Parch** = 23.91% 
  - **Fare** = 13.02%
  - **SibSp** = 5.16%
  - **Age** = 1.23%


### 4.5 - Create Features
- **Sex:** Will be given numerical values for their respective categories
- **Cabin:** Will be given numerical values for their respective categories and be grouped by the first letter
- **Cabin:** Will receive numerical values for their respective categories and will be grouped by the first letter. Missing values will be set to "N"
- **Name:** I believe the only use here would be to take the titles and insert numerical values into them
- **Boarders:** They will receive numerical values for their respective categories

### 4.6 - Drop Features
- **Ticket:** There is a high degree of variation and dispersion in this data. I don't see much use in them, maybe group them together in some way and check for some relationship with the tariff


### 4.7 - Normalization and Standardization
- The decisive factor in rescaling the data is the **exclusive use of the KNN model.** It is extremely sensitive to **discrepant scales, outliers and prefers normalized data.**
- There is a need to rescale the data because the range of the data goes from **0 to 512.33**
- Most numerical data does not follow a normal distribution. This corroborates the fact that data must be standardized in order to bring it as close as possible to its original state.
- **But focusing on the best use of the KNN model, I will use normalization.**

## 5 - Machine Learning Exploration

### 5.1 - With Outliers Dataset
- The model performances on the test dataset seem to decrease slightly as the number of neighbors increases.
- The F1 Score, which balances precision and recall, ranges from **0.774292** to **0.795999** in train dataset and ranges from **0.825528** to **0.866613** in test dataset.
- On the training dataset, the scores are higher, indicating potential overfitting, especially with a smaller number of neighbors.


### 5.2 - Without Outliers Dataset
- The model performances on the test dataset are generally better compared to the dataset with outliers.
- The F1 Score, which balances precision and recall, ranges from **0.827385** to **0.828747** in train dataset and ranges from **0.816357** to **0.841127** in test dataset.
- On the training dataset, the scores are still relatively high, suggesting good performance, and they are closer to the test scores compared to the dataset with outliers.


### 5.3 - Neighbor Selection
- It seems that using 5 neighbors gives good results for both datasets (e.g., 0.789238 to 0.826667 for the test dataset).
- Selecting an appropriate number of neighbors is crucial; too few neighbors might lead to overfitting, while too many might result in underfitting.


### 5.4 - Effect of Outliers
- It seems that using 5 neighbors of without outliers gives good results for both datasets (e.g., **0.833809** to **0.838452** for the test dataset).
- Selecting an appropriate number of neighbors is crucial; too few neighbors might lead to overfitting, while too many might result in underfitting.

## 6 - Submissions Results
After optimizing the model with the **GridSearchCV** algorithm, the best KNN submission results were:

| Model                  | Precision | Recall | F1-Score | Accuracy | Mean - CV | Std - CV | Range - CV    | Kaggle Result |
| ---------------------- | ----------| ------ | -------- | -------- | --------- | -------- | --------------| ------------- |
| KNN - Without Outliers | 78.34     | 69.10  | 73.43    | 85.12    | 81.28     | 6.50     | [74.78, 87.77]| 0.76555       |
| KNN - With    Outliers | 83.54     | 78.65  | 81.02    | 85.86    | 81.71     | 5.04     | [76.67, 86.75]| 0.74880       |


## 7 - Model Comparison
In order to understand the model's performance and how it behaved, I compared knn's results with those of the Gradient Boosting Classifier model.

| Model                  | Precision | Recall | F1-Score | Accuracy | Mean - CV | Std - CV | Range - CV    | Kaggle Result |
| ---------------------- | ----------| ------ | -------- | -------- | --------- | -------- | --------------| ------------- |
| KNN - Without Outliers | 78.34     | 69.10  | 73.43    | 85.12    | 81.28     | 6.50     | [74.78, 87.77]| 0.76555       |
| KNN - With    Outliers | 83.54     | 78.65  | 81.02    | 85.86    | 81.71     | 5.04     | [76.67, 86.75]| 0.74880       |
| GBC - Without Outliers | 88.24     | 75.84  | 81.57    | 89.80    | 81.26     | 4.43     | [76.83, 85.69]| 0.76076       |
| GBC - With    Outliers | 95.58     | 88.60  | 91.96    | 94.05    | 83.51     | 5.24     | [78.27, 88.75]| 0.72727       |


## 8 - Conclusion

Although the Gradient Boosting Classifier (GBC) is often considered more robust than K-Nearest Neighbors (KNN), in this specific context, **KNN has shown remarkable adaptability to Kaggle's evaluation criteria (0.7655 x 0.76076)**. This effectiveness can be attributed to carefully adjusted transformations in the original dataset, optimized specifically for the KNN algorithm.

It is worth noting that among the evaluated models, only KNN, **without the presence of outliers**, produced Kaggle results consistent with the cross-validation range. This emphasizes the importance of data transformations specific to KNN. Interestingly, the inclusion of outliers in the KNN model resulted in Kaggle performance beyond expectations, reinforcing the idea that the presence of outliers can significantly impact KNN performance, possibly explaining the observed discrepancy in this case.

Another point to consider in explaining KNN's superior performance compared to GBC in Kaggle metrics is the possibility of overfitting in GBC. It is observed that in GBC, metrics such as **Precision, Recall, F1-Score, and Accuracy** were higher than those of KNN, indicating that GBC may not have generalized well to data beyond that used in training.

In conclusion, while GBC is recognized as a powerful model, the highlighted performance of KNN on the Kaggle platform underscores the importance of careful data preprocessing and consideration of specific dataset characteristics to achieve optimized model performance.
