import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('random_forest_model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Define ethnic group options
ethnicity_options = [
    'Hispanic', 'Latino', 'Native Indian', 'Others', 'Pacifica',
    'White European', 'asian', 'black', 'middle eastern', 'mixed', 'south asian'
]

# Streamlit App
st.title("Autism Spectrum Disorder Predictor")
st.markdown("Answer the following questions to assess potential autism spectrum disorder (ASD) risk.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

eth = st.selectbox("Ethnicity", ethnicity_options)

jaun = st.selectbox("Has the child experienced jaundice?", ["No", "Yes"])
jaun_val = 1 if jaun == "Yes" else 0

q_values = []
for i in range(11):
    answer = st.selectbox(f"Q{i+1}", ["No", "Yes"], key=f"q{i}")
    q_values.append(1 if answer == "Yes" else 0)

text2 = st.text_input("Enter relation to the child")

# Predict button
if st.button("Predict"):
    try:
        # Convert ethnicity to one-hot encoding
        eth_dict = {ethn: 0 for ethn in ethnicity_options}
        eth_dict[eth] = 1
        eth_encoded = list(eth_dict.values())

        # Combine all features
        features = q_values[:10] + [age, sex_val, jaun_val, q_values[10]] + eth_encoded

        # Prediction
        prediction = model.predict([features])[0]

        if prediction == 1:
            st.error("🧠 Your child may have Autism Spectrum Disorder. We recommend consulting a specialist.")
        else:
            st.success("✅ Your child appears to be typically developing.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
