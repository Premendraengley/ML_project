# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
url = "C:/Users/Lenovo/Desktop/Diabities_Prediction_Project/diabetes_prediction_dataset.csv"
column_names = ["blood_glucose", "bmi", "diabetes", "age"]  # Add the correct column names
dataset = pd.read_csv(url, header=None, names=column_names, skiprows=1, na_values=['No Info'])  # Skip rows with non-numeric values

# Display the first few rows of the dataset
print(dataset.head())

# Drop rows with NaN values
dataset = dataset.dropna()

# Separate features and target variable
X = dataset.iloc[:, :-1]
y = dataset["diabetes"]  # Assuming 'diabetes' is the target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Visualize the feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(8).plot(kind='barh')
plt.title("Feature Importances")
plt.show()
