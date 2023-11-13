# KNN Classification


This project aims to utilize the K-Nearest Neighbors (KNN) algorithm for data classification, primarily focusing on identifying survivors of the Titanic shipwreck. The approach encompasses an initial phase of exploratory analysis, followed by the application of KNN, the use of validation techniques, and ultimately, hyperparameter optimization. This comprehensive approach will enable a deeper and more effective understanding of the survival prediction task in the context of the Titanic dataset.


![vist Card](https://cdn.discordapp.com/attachments/1173750582092251139/1173750628049223680/462214b129e4860466cee98ab50b3793.jpg?ex=65651774&is=6552a274&hm=9b132593f08df3d629cc2c9db5c0afa627d69a5f38abca13cbd79a11415d7d3d&)

  

## Table of Contents

- [1. Introduction](#1-Introduction)

- [2. Data Understanding](#2-Data-Understanding)

- [3. Action Plan](#3-Action-Plan)

- [4. Data Insights](#4-Data-Insights)

- [5. Machine Learning Metrics](#5-Machine-Learning-Metrics)

- [6. Results](#6-Results)

- [7. Conclusion](#5-Conclusion)
  
  

## 1. Introduction
This project aims to conduct an **exploratory** analysis of the Titanic data set and employ the **KNN (K-Nearest Neighbors)** model to classify the survivors of the shipwreck. The widely recognized **Titanic** dataset is available at **[Kaggle](https://www.kaggle.com/c/titanic/data)**, **[Stanford](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)** and also at **[Department of Biostatistics](https://hbiostat.org/data/)**. Despite being a simple dataset, designed for studies and focused on solving a single question, it is possible to apply advanced data analysis techniques, extract insights and knowledge in a similar way to what would be done in more complex and challenging datasets.
  

## 2. Data Understanding

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


## 3. Action Plan

  

### 3.1. Objective

After completing the project of exploratory analysis and implementation of [KNN in the Wine database](https://github.com/lfaferreira/knn-wine), I decided to further deepen the use of this model by incorporating more sophisticated techniques. This repository aims to explore the capabilities of the KNN algorithm, focusing on optimization and enhancement to solve the Titanic problem.

  

### 3.2. Tools and Frameworks

Scope of tools used in the project:

  

- Python 3.11.5

- Jupyter Notebook

- Git & GitHub

- Kaggle

- Pandas

- Numpy

- Machine Learning Classification Model (KNN)

  
## 4. Data Insights
loading...

## 5. Machine Learning Metrics
loading...

## 6. Results
loading...

## 7. Conclusion
loading...