# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st


# Load the dataset
data = pd.read_csv("path/Downloads/creditcard.csv")

# (Assuming 'Class' column is the target variable indicating fraud or not)
# Check dimensions of the dataset
print("Dimensions of the dataset:", data.shape)


# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())


# Check for missing values
print("Missing values in the dataset:")
print(data.isnull().sum())


# Explore the distribution of the target variable
print("Distribution of the target variable ('Class'):")
print(data['Class'].value_counts())


#Percentage of Legit and Frauds before sampling
print("Percentage of data belongs to Legit:", data["Class"].value_counts()[0]*100/len(data))
print("Percentage of data belongs to frauds:", data["Class"].value_counts()[1]*100/len(data))


#Imbalance Dataset Bar chart
sns.countplot(x="Class", data = data)
plt.title('UnEqually Distributed Classes', fontsize=14)
plt.show()


# Compute the correlation matrix before sampling
correlation_matrix = data.corr()
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Unbalanced Dataset Correlation Matrix Heatmap')
plt.show()


#Describe dataset
data.describe()


#Legal and Fraud count
legit = data[data['Class']==0]
fraud = data[data['Class']==1]

#Fraud Dimension
fraud.shape


#Legit Dimension
legit.shape

# Sampling
legit_sample = legit.sample(n=len(fraud),random_state=2)
data = pd.concat([legit_sample,fraud])


#Percentage of Legit and Frauds after sampling
print("Percentage of data belongs to Legit:", data["Class"].value_counts()[0]*100/len(data))
print("Percentage of data belongs to frauds:", data["Class"].value_counts()[1]*100/len(data))


#Balanced dataset Bar chart
sns.countplot(x="Class", data = data)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# Compute the correlation matrix after sampling
correlation_matrix = data.corr()
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Balanced Correlation Matrix Heatmap')
plt.show()


# Splitting features and target variable
X = data.drop('Class', axis=1)
y = data['Class']


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initializing classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier()
}


# Training and evaluating classifiers
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

  # Calculating performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Performance Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("------------------------------")


# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")











