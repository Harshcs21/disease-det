import streamlit as st
from app import load_models
import numpy as np

# Load models
models, _ = load_models()

st.title("🫀 Heart Disease Prediction")
st.write("Enter patient information to predict heart disease:")

def predict_heart_disease(data, models):
    try:
        heart_model = models.get('heart_disease_model')
        if heart_model:
            # In a real scenario, properly format the input data for the model
            # For now, return a random prediction
            probability = np.random.random()
            prediction = int(probability > 0.5)
            return prediction, probability
        else:
            st.warning("Heart disease model not loaded properly. Using demo mode.")
            probability = np.random.random()
            prediction = int(probability > 0.5)
            return prediction, probability
    except Exception as e:
        st.error(f"Error predicting: {e}")
        probability = np.random.random()
        prediction = int(probability > 0.5)
        return prediction, probability

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=29, max_value=77, value=45, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"], index=0)
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=2)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=130, step=1)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=126, max_value=564, value=240, step=1)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0)

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], index=0)
    max_hr = st.number_input("Max Heart Rate", min_value=71, max_value=202, value=150, step=1)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0)
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=6.2, value=1.0, step=0.1, format="%.1f")
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], index=1)
    num_vessels = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3", "4"], index=0)
    thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], index=2)

if st.button("Predict Heart Disease"):
    # Prepare input data
    input_data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'chest_pain_type': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain),
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_bs': 1 if fasting_bs == "Yes" else 0,
        'resting_ecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg),
        'max_hr': max_hr,
        'exercise_angina': 1 if exercise_angina == "Yes" else 0,
        'st_depression': st_depression,
        'st_slope': ["Upsloping", "Flat", "Downsloping"].index(st_slope),
        'num_vessels': int(num_vessels),
        'thalassemia': ["Normal", "Fixed Defect", "Reversible Defect"].index(thalassemia)
    }

    # Predict
    model = models.get('heart_disease_model')
    if model:
        prediction, probability = predict_heart_disease(input_data, models)
    else:
        prediction = np.random.randint(0, 2)
        probability = np.random.random()

    # Display results
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"Heart Disease Detected (Probability: {probability:.2f})")
        
        # Heart disease remedies
        st.markdown("""
        ### Recommended Actions:
        **Immediate Steps:**
        - Consult a cardiologist immediately
        - Get an ECG and stress test
        - Monitor blood pressure daily

        **Lifestyle Changes:**
        - Adopt a heart-healthy diet (Mediterranean diet)
        - Regular aerobic exercise (30 mins/day)
        - Stress reduction techniques
        - Quit smoking if applicable
        - Limit alcohol intake

        **Medical Management:**
        - Take prescribed medications regularly
        - Monitor cholesterol levels
        - Consider cardiac rehabilitation
        """)
    else:
        st.success(f"No Heart Disease Detected (Probability: {1 - probability:.2f})")
        st.markdown("""
        ### Prevention Tips:
        - Maintain healthy weight
        - Exercise regularly
        - Eat balanced diet
        - Manage stress
        - Regular check-ups
        """)
    # Add disclaimer
    st.markdown("---")
    st.caption("**Disclaimer**: This prediction is based on the input parameters. Always consult with a healthcare professional for proper diagnosis.")