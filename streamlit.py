import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
heart_disease_model = pickle.load(open('heart_disease.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",)
st.header("Patient Information")
st.sidebar.header("Heart Disease Prediction App")
st.markdown("""
    <style>
        .main {background-color: #DCD7C9;}
        .stTextInput, .stNumberInput {border-radius: 10px;}
        .stButton>button {background-color: #AA60C8; color: white; border-radius: 10px;}
    </style> 
            """, unsafe_allow_html=True)


Age = st.number_input("Age")
sex = st.number_input("Sex")
cp = st.number_input("Chest Pain Type (cp)")
trestbps = st.number_input("Resting Blood Pressure (trestbps)")
chol = st.number_input("Cholesterol (chol)")
fbs = st.number_input("Fasting Blood Sugar (fbs)")
restecg = st.number_input("Resting ECG (restecg)")
thalach = st.number_input("Maximum Heart Rate (thalach)")
exang = st.number_input("Exercise-Induced Angina (exang)")
oldpeak = st.number_input("ST Depression (oldpeak)")
slope = st.number_input("Slope of Peak Exercise ST Segment (slope)")
ca = st.number_input("Number of Major Vessels (ca)")
thal = st.number_input("Thalassemia (thal)")


# Code for prediction
#heart_disease = ''

# Prediction
if st.button("Get Results"):
    # Create a NumPy array from the inputs
    input_data = np.array([[Age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    
    # Apply the scaler transformation
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    heart_pred = heart_disease_model.predict(input_data_scaled)

    # Display the result    
    if heart_pred[0] == 0:
        st.markdown("<h3 style='color: green; text-align: center;'>✅ The person does NOT have heart disease.</h3>", unsafe_allow_html=True)

    else:
        st.markdown("<h3 style='color: red; text-align: center;'>❌ The person HAS heart disease.</h3>", unsafe_allow_html=True)
       