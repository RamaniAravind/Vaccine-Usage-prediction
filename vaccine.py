import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
vaccine = pd.read_csv('h1n1_vaccine_prediction.csv')

# Preprocessing: Fill missing values
vaccine.fillna(vaccine.mode().iloc[0], inplace=True)  # Filling missing values with mode

# Define features and target variable
X = vaccine.drop(columns=['h1n1_vaccine'])
y = vaccine['h1n1_vaccine']

# Encode categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit app
st.title('H1N1 Vaccine Prediction')

# Display the virus logo
virus_logo = Image.open("virus.JPEG")
st.image(virus_logo, width=500, use_column_width=False)

# Sidebar for user input
st.sidebar.title("Predict Vaccine Acceptance")

# Collect user input for prediction
inputs = {}
for col in X.columns:
    if col in ['qualification', 'marital_status', 'employment', 'income_level', 'housing_status']:
        inputs[col] = st.sidebar.selectbox(col, vaccine[col].unique())
    else:
        inputs[col] = st.sidebar.radio(f"{col} (0=NO, 1=YES)", [0, 1])

# Convert input into a DataFrame
input_data = pd.DataFrame([inputs])

# Encode categorical variables in input data
input_data = pd.get_dummies(input_data, drop_first=True)

# Make prediction
if st.sidebar.button('Predict'):
    # Ensure input_data has the correct columns and is in the correct format
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    
    st.write(f"Prediction: {'YES' if prediction[0] == 1 else 'NO'}")
    st.write(f"Probability of YES: {proba[0][1]:.2f}")

# Data Visualization
st.write("## Data Visualization")

# Visualization Settings
plot_type = st.sidebar.selectbox("Select plot type", ["Bar Plot", "Pie Chart", "Count Plot"])
feature = st.sidebar.selectbox("Select feature to visualize", X.columns)

# Visualizations based on user selection
if plot_type == "Bar Plot":
    st.bar_chart(vaccine[feature].value_counts())
elif plot_type == "Pie Chart":
    fig, ax = plt.subplots()
    vaccine[feature].value_counts().plot(kind='pie', autopct='%0.2f%%', ax=ax)
    st.pyplot(fig)
elif plot_type == "Count Plot":
    fig, ax = plt.subplots()
    sns.countplot(x=feature, data=vaccine, hue='h1n1_vaccine', ax=ax)
    st.pyplot(fig)

# Data Overview
st.write("## Data Overview")
st.write(vaccine.head())
