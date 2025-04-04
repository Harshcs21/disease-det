import streamlit as st
from app import load_models
import numpy as np
import pandas as pd

# Load models
models, _ = load_models()

# Define the column names for the diabetes dataset
# This is needed since X.columns is not defined in this file
DIABETES_COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def predict(model, input_data):
    """
    Make predictions using the loaded model.

    Parameters:
    - model: The loaded Random Forest model
    - input_data: Dictionary or DataFrame with feature values

    Returns:
    - Prediction result (0 or 1)
    """
    if isinstance(input_data, dict):
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        # Ensure all features are present in the correct order
        input_df = input_df.reindex(columns=DIABETES_COLUMNS, fill_value=0)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.reindex(columns=DIABETES_COLUMNS, fill_value=0)
    else:
        raise TypeError("Input must be a dictionary or DataFrame")

    return model.predict(input_df)[0]

st.title("🩸 Diabetes Prediction")
st.write("Enter patient information to predict diabetes:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=3, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=122, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=79, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.4, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42, value=0.47, step=0.01, format="%.2f")
    age = st.number_input("Age (years)", min_value=21, max_value=81, value=33, step=1)

if st.button("Predict Diabetes"):
    # Prepare input data
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    # Predict
    diabetes_model = models.get('diabetes_model')
    
    if diabetes_model is None:
        st.error("Error: Diabetes model not found. Please check your model loading function.")
    else:
        # Get prediction
        prediction = predict(diabetes_model, input_data)
        
        # Get probabilities
        input_values = [list(input_data.values())]
        probabilities = diabetes_model.predict_proba(input_values)[0]
        probability = probabilities[1]  # Index 1 is for class 1 (diabetic)
        
        # Display results
        st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"Diabetic (Probability: {probability:.2f})")
        
        # Diabetes remedies
        st.markdown("""
        ### Diabetes Management:
        **Dietary Recommendations:**
        - Low glycemic index foods
        - High fiber vegetables
        - Lean proteins
        - Healthy fats (avocados, nuts)
        - Limit processed carbs/sugars

        **Lifestyle Changes:**
        - Regular exercise (150 mins/week)
        - Weight management
        - Stress reduction
        - Quality sleep (7-8 hours)

        **Medical Management:**
        - Monitor blood sugar regularly
        - Take prescribed medications
        - Regular HbA1c checks
        - Foot care examinations
        """)
    else:
        st.success(f"Non-Diabetic (Probability: {1 - probability:.2f}")
        st.markdown("""
        ### Prevention Tips:
        - Maintain healthy weight
        - Exercise regularly
        - Balanced diet
        - Limit sugary foods/drinks
        - Regular check-ups
        """)
        
        # Add disclaimer
        st.markdown("---")
        st.caption("**Disclaimer**: This prediction is based on the input parameters. Always consult with a healthcare professional for proper diagnosis.")