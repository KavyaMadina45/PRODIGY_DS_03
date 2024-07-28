# Bank Marketing Data Analysis and Prediction

This project involves analyzing a bank dataset and building a decision tree model to predict the balance based on various features. The analysis includes data preprocessing, handling missing values, encoding categorical variables, splitting the dataset into training and testing sets, and evaluating the decision tree model.

## Project Description

The project demonstrates how to:
1. Load and inspect the bank dataset.
2. Clean the dataset by removing NaN values and duplicates.
3. Encode categorical variables.
4. Split the data into training and testing sets.
5. Build a decision tree model using scikit-learn.
6. Evaluate the model's performance using accuracy, confusion matrix, and classification report.

## Installation Instructions

To run this project, you need to have Python installed along with the following libraries:
- pandas
- scikit-learn
- seaborn
- matplotlib
- graphviz
- pydotplus

You can install these libraries using pip:
```sh
pip install pandas scikit-learn seaborn matplotlib graphviz pydotplus
```
**Usage**
1) Place your dataset file (bank.csv) in an accessible directory.
2) Update the file path in the code to match the location of your dataset file.
3) Run the script to perform the analysis and build the decision tree model.

**Example**
```sh
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from IPython.display import Image

# Load the dataset
url = "/home/kavya/Downloads/bank.csv"
data = pd.read_csv(url)

# Step 2: Preprocess the data
# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Separate features and target variable
X = data.drop('balance', axis=1)
y = data['balance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = clf.predict(X_test)
print(y_pred)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
```
**Features**
* Data Loading: Efficiently loads the bank dataset from a CSV file.
* Data Cleaning: Removes NaN values and duplicates from the dataset.
* Data Encoding: Encodes categorical variables to numerical values.
* Data Splitting: Splits the dataset into training and testing sets.
* Model Building: Builds a decision tree model to predict the balance.
* Model Evaluation: Evaluates the model's performance using accuracy, confusion matrix, and classification report.

**Contributing**

If you want to contribute to this project, please follow these steps:

1) Fork the repository.
2) Create a new branch (git checkout -b feature-branch).
3) Make your changes.
4) Commit your changes (git commit -m 'Add new feature').
5) Push to the branch (git push origin feature-branch).
6) Create a new Pull Request.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Contact Information**
For any questions or issues, please contact  Kavya at madinakavya6@gmail.com
