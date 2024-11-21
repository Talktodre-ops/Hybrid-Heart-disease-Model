#importing the necessary modules 


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Loading the pre-trained hybrid model
with open('hybrid_model.pkl', 'rb') as f:
    hybrid_model = pickle.load(f)

# Extracting the components of the hybrid model
base_models = hybrid_model['base_models']
meta_model = hybrid_model['meta_model']
scaler = hybrid_model['scaler']

#the Feature names used in training the base models and the hybrid models
feature_names = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity',
                 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'GenHlth',
                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

#This is the Streamlit app title
st.title('Heart Disease Prediction')

#this is for user data privacy
st.sidebar.markdown("""
**Disclaimer:**
* User data is used solely for generating a prediction and is not stored or shared.
* The model used in this application is trained on anonymized data.
* The results provided are intended for informational purposes only and should not replace professional medical advice.
""")


#the Sidebar for user input
st.sidebar.header('Input Features')

# Defining the features 
feature_definitions = {
    'HighBP': {'type': 'binary', 'label': 'Do you have High Blood Pressure or Have you ever had High Blood Pressure?', 'options': ['No', 'Yes']},
    'HighChol': {'type': 'binary', 'label': 'Do you high Cholesterol level?', 'options': ['No', 'Yes']},
    'BMI': {'type': 'numeric', 'label': 'what is your body mass index (BMI)', 'min': 0, 'max': 80},
    'Smoker': {'type': 'binary', 'label': 'Do you smoke?', 'options': ['No', 'Yes']},
    'Stroke': {'type': 'binary', 'label': 'Have you ever had Stroke?', 'options': ['No', 'Yes']},
    'Diabetes': {'type': 'binary', 'label': 'Are you Diabetic or have you ever been diabetic?', 'options': ['No', 'Yes']},
    'PhysActivity': {'type': 'binary', 'label': 'Are you Physically Active?', 'options': ['No', 'Yes']},
    'Fruits': {'type': 'binary', 'label': 'Do you Consumes Fruits', 'options': ['Yes', 'No']},
    'Veggies': {'type': 'binary', 'label': 'Do you Consumes Veggies', 'options': ['No', 'Yes']},
    'HvyAlcoholConsump': {'type': 'binary', 'label': 'Do you consume Heavy amount of Alcohol', 'options': ['No', 'Yes']},
    'AnyHealthcare': {'type': 'binary', 'label': 'Do you have any Healthcare coverage?', 'options': ['No', 'Yes']},
    'GenHlth': {'type': 'numeric', 'label': 'On a scale of 1 to 5, rate your General Health', 'min': 0, 'max': 5},
    'MentHlth': {'type': 'numeric', 'label': 'In the past month, how many days have you felt mentally unwell?', 'min': 0, 'max': 30},
    'PhysHlth': {'type': 'numeric', 'label': 'In the past month, How many days have you felt physically unwell?', 'min': 0, 'max': 30},
    'DiffWalk': {'type': 'binary', 'label': 'Do you have difficulty walking', 'options': ['No', 'Yes']},
    'Sex': {'type': 'binary', 'label': 'Are you a Male or a Female?', 'options': ['Male', 'Female']},
    'Age': {'type': 'numeric', 'label': 'How old are you?', 'min': 13, 'max': 100}
}

# Geting input from user in the sidebar for the pre-defined features
feature_values = []
for feature, definition in feature_definitions.items():
    if definition['type'] == 'binary':
        value = st.sidebar.radio(definition['label'], options=definition['options'])
        feature_values.append(1 if value == 'Yes' else 0)
    elif definition['type'] == 'numeric':
        value = st.sidebar.slider(definition['label'], definition['min'], definition['max'], (definition['min'] + definition['max']) // 2)
        feature_values.append(value)
    else:
        # Handling other feature types
        value = st.sidebar.text_input(definition['label'])
        feature_values.append(value)

# Converting input data to DataFrame so as to be able to use them for prediction
new_data_df = pd.DataFrame([feature_values], columns=feature_names)

# Scaling the new data point
new_data_scaled = scaler.transform(new_data_df)

# Generating meta-features using base models
meta_features_new = np.zeros((1, len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    meta_features_new[0, i] = model.predict_proba(new_data_scaled)[0, 1]  # Use predict_proba for probability

# Predicting the probability using the meta-model
probability = meta_model.predict_proba(meta_features_new)[0, 1]

# Determine the risk level based on probability
risk_level = "Low" if 5 *(probability * 100) < 30 else ("Moderate" if 5 *(probability * 100) < 70 else "High")

# Display the prediction result
st.subheader('Prediction Result:')
st.write(f"Risk Level: {risk_level}")
st.write(f"Probability of heart disease: {5 *(probability * 100):.2f}%")

# Display the prediction result
st.subheader('Prediction Result:')
st.write(f"Probability of heart disease: {5 *(probability * 100):.2f}%")

# Show a pie chart of the prediction result
st.subheader('Prediction Probability:')
labels = 'No Heart Disease', 'Heart Disease'
sizes = [100 - 5 * (probability * 100), 5 * (probability * 100)]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Show feature importance for Random Forest (if available)
if 'RT' in base_models:
    st.subheader('Feature Importance:')
    importances = base_models['RT'].feature_importances_
    if len(importances) == len(feature_names):
        indices = np.argsort(importances)[::-1]
        fig2, ax2 = plt.subplots()
        sns.barplot(y=[feature_names[i] for i in indices], x=importances[indices], ax=ax2)
        st.pyplot(fig2)
    else:
        st.error("Error: The number of features does not match the model's expectations.")
        
   #End.
