# Diabetes Prediction Project (ML)
Overview
This project aims to develop a machine learning model to predict the likelihood of diabetes in patients based on various medical attributes. The goal is to use this model to assist healthcare professionals in identifying high-risk individuals for further screening and early intervention.

Table of Contents
Overview
Dataset
Installation
Usage
Model Training
Results
Contributing
License
Acknowledgments
Dataset
The dataset used for this project is the Pima Indians Diabetes Database, available from the UCI Machine Learning Repository. It contains medical data for 768 female patients of Pima Indian heritage, aged 21 and older. The dataset includes the following features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration (mg/dL)
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skinfold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/mL)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1) indicating whether the patient has diabetes
Installation
To run this project, you will need Python 3.x and the following libraries:

NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn

Model Training
The project includes a Jupyter Notebook (diabetes_prediction.ipynb) that details the entire process of training the model, including:

Data Preprocessing: Handling missing values, feature scaling, and splitting the dataset into training and testing sets.
Exploratory Data Analysis (EDA): Visualizing the distribution of features and their relationships with the target variable.
Model Selection: Trying different algorithms (e.g., Logistic Regression, Random Forest, SVM) and selecting the best performing model based on evaluation metrics.
Hyperparameter Tuning: Using Grid Search or Random Search to find the optimal hyperparameters for the chosen model.
Evaluation: Assessing the model's performance using metrics such as accuracy, precision, recall, and ROC-AUC.
Results
The final model achieved the following performance on the test set:

Accuracy: 78%
Precision: 0.75
Recall: 0.70
ROC-AUC: 0.82
