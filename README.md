# KNN Classification

This project aims to utilize the K-Nearest Neighbors (KNN) algorithm for data classification, primarily focusing on identifying survivors of the Titanic shipwreck. The approach encompasses an initial phase of exploratory analysis, followed by the application of KNN, the use of validation techniques, and ultimately, hyperparameter optimization. This comprehensive approach will enable a deeper and more effective understanding of the survival prediction task in the context of the Titanic dataset.


![vist Card](https://cdn.discordapp.com/attachments/1173750582092251139/1175485464438907111/titanic.webp?ex=656b6726&is=6558f226&hm=96ea5d19025c1c1a4982cf0a8a8964f7d8fd94711f1d9009a6af9e362bc83d63&)


## Table of Contents

- [1 - Introduction](#1-Introduction)

- [2 - Data Understanding](#2-Data-Understanding)

- [3 - Action Plan](#3-Action-Plan)

- [4 - Exploratory Data Analysis](#4-Exploratory-Data-Analysis)

- [5 - Machine Learning Exploration](#5-Machine-Learning-Exploration)

- [6 - Submissions Results](#6-Submissions-Results)

- [7 - Models Comparison](#7-Models-Comparison)

- [8 - Conclusion](#8-Conclusion)
  
  

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

- Machine Learning Classification Model (KNN)

  
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


### 4.6 - Drop Features
- **Name:** I believe that the only use here would be to take the titles and input numerical values into them
- **Ticket:** There is a high degree of variation and dispersion in this data. I don't see much use in them, maybe group them together in some way and check for some relationship with the tariff
- **Embarked:** The missing values


### 4.7 - Normalization and Standardization
- The decisive factor in rescaling the data is the **exclusive use of the KNN model.** It is extremely sensitive to **discrepant scales, outliers and prefers normalized data.**
- There is a need to rescale the data because the range of the data goes from **0 to 512.33**
- Most numerical data does not follow a normal distribution. This corroborates the fact that data must be standardized in order to bring it as close as possible to its original state.
- **But focusing on the best use of the KNN model, I will use normalization.**

## 5 - Machine Learning Exploration

### 5.1 - With Outliers Dataset
- The model performances on the test dataset seem to decrease slightly as the number of neighbors increases.
- The F1 Score, which balances precision and recall, ranges from 0.775453 to 0.796534.
- On the training dataset, the scores are higher, indicating potential overfitting, especially with a smaller number of neighbors.


### 5.2 - Without Outliers Dataset
- The model performances on the test dataset are generally better compared to the dataset with outliers.
- The F1 Score ranges from 0.816000 to 0.821941, indicating better balance between precision and recall.
- On the training dataset, the scores are still relatively high, suggesting good performance, and they are closer to the test scores compared to the dataset with outliers.


### 5.3 - Neighbor Selection
- It seems that using 5 neighbors gives good results for both datasets (e.g., 0.789238 to 0.826667 for the test dataset).
- Selecting an appropriate number of neighbors is crucial; too few neighbors might lead to overfitting, while too many might result in underfitting.


### 5.4 - Effect of Outliers
- The comparison between the results with and without outliers suggests that removing outliers improves the model performance, particularly in terms of F1 Score.


## 6 - Submissions Results

|Submissions|Kaggle Submissions Score|Top Percentage|Position      |Accuracy Score|Train Score|Train Precision|Train Recall|Train F1  |
|-----------|------------------------|--------------|--------------|--------------|-----------|---------------|------------|----------|
|1st        |0.74401                 |Top 90%       |13768 of 15348|82.67%        |85.04%     |84.75%         |85.05%      |84.71%    |
|2nd        |0.76076                 |Top 83%       |12639 of 15348|83.33%        |84.82%     |84.51%         |84.82%      |84.47%    |
|3rd        |0.76555                 |Top 79%       |12010 of 15362|83.33%        |84.82%     |84.51%         |84.82%      |84.47%    |


### 6.1 - Observations
- The **first** and **second** subimissions were training the final model differently from the third onwards. In this case, I wasn't using train_test_split to train the model that would make the final prediction and I was training the model with the complete base. That's why the second and third submissions have the same model status but different placements in Kaggle.


## 7 - Model Comparison
| Model                           | Accuracy Score | Model Score |
|---------------------------------|-----------------|-------------|
| Logistic Regression             | 84.0%           | 80.36%      |
| Support Vector Machines         | 83.33%          | 81.7%       |
| KNN                             | 83.33%          | 84.82%      |
| Linear SVC                      | 82.67%          | 80.58%      |
| Random Forest                   | 82.0%           | 98.44%      |
| Stochastic Gradient Descent     | 82.0%           | 81.03%      |
| Naive Bayes                     | 80.67%          | 77.68%      |
| Gradient Boosting Classifier    | 80.0%           | 91.07%      |
| Perceptron                       | 78.67%          | 79.24%      |
| Decision Tree                    | 72.0%           | 98.44%      |



## 8 - Conclusion
loading...
