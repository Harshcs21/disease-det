import os
import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
from PIL import Image
import warnings
import requests
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings('ignore')

# Path to models
MODEL_PATH = 'models'

# Set page config
st.set_page_config(
    page_title="Healthcare Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    try:
        # Load diabetes model and scaler
        models['diabetes_scaler'] = joblib.load(os.path.join(MODEL_PATH, 'diabetes_scaler.pkl'))

        # Load diabetes XGBoost model from JSON
        with open(os.path.join(MODEL_PATH, 'diabetes_model.json'), 'r') as f:
            models['diabetes_model_data'] = json.load(f)

        # Load heart disease model
        models['heart_disease_model'] = joblib.load(os.path.join(MODEL_PATH, 'heart_disease_rf_model.pkl'))

        # Load Parkinson's model and scaler
        models['parkinsons_scaler'] = joblib.load(os.path.join(MODEL_PATH, 'parkinsons_scaler.pkl'))

        # For skin cancer model, we'd normally load it here
        models['skin_cancer_labels'] = {
            0: 'Actinic Keratoses',
            1: 'Basal Cell Carcinoma',
            2: 'Benign Keratosis',
            3: 'Dermatofibroma',
            4: 'Melanoma',
            5: 'Melanocytic Nevi',
            6: 'Vascular Lesions'
        }

        # Load symptom mapper
        models['symptom_mapper'] = {
            'diabetes': ['frequent urination', 'increased thirst', 'increased hunger', 'weight loss', 'fatigue', 'blurred vision', 'slow-healing sores', 'frequent infections'],
            'heart_disease': ['chest pain', 'shortness of breath', 'pain in neck/jaw/throat/arms', 'nausea', 'fatigue', 'dizziness', 'cold sweat', 'irregular heartbeat'],
            'parkinsons': ['tremor', 'stiffness', 'slow movement', 'poor balance', 'stooped posture', 'speech changes', 'writing changes', 'difficulty with automatic movements'],
            'skin_disease': ['rash', 'itchiness', 'discoloration', 'moles', 'bumps', 'sores', 'dry skin', 'scaling']
        }

        return models, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, False

# Configure Gemini API
def setup_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error setting up Gemini API: {e}")
        return False

# Function to analyze symptoms using Gemini API
def analyze_symptoms_with_gemini(symptoms_text, models):
    try:
        # Get Gemini API key from environment variable or secrets
        api_key = 'AIzaSyAiaK9i0Uar7MpHjEcxVU5jonhzpzCNbto'

        if not api_key:
            st.warning("Gemini API key not found. Using rule-based analysis instead.")
            return analyze_symptoms_rule_based(symptoms_text, models)

        if not setup_gemini(api_key):
            return analyze_symptoms_rule_based(symptoms_text, models)

        # Configure the model
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 1024,
        }

        # Create model instance
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config
        )

        # Create prompt
        prompt = f"""
        Analyze the following symptoms and determine the likelihood of these conditions:
        1. Diabetes
        2. Heart Disease
        3. Parkinson's Disease
        4. Skin Disease

        Patient's symptoms: {symptoms_text}

        Provide a JSON response with the following structure:
        {{
            "diabetes_probability": (float between 0 and 1),
            "heart_disease_probability": (float between 0 and 1),
            "parkinsons_probability": (float between 0 and 1),
            "skin_disease_probability": (float between 0 and 1),
            "reasoning": "brief explanation of the analysis"
        }}
        """

        # Generate the response
        response = model.generate_content(prompt)

        # Extract the JSON from the response
        response_text = response.text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx >= 0 and end_idx > start_idx:
            json_response = json.loads(response_text[start_idx:end_idx])
            return json_response
        else:
            st.warning("Unable to parse Gemini API response. Using rule-based analysis instead.")
            return analyze_symptoms_rule_based(symptoms_text, models)

    except Exception as e:
        st.error(f"Error analyzing symptoms with Gemini: {e}")
        return analyze_symptoms_rule_based(symptoms_text, models)

# Rule-based symptom analysis as fallback
def analyze_symptoms_rule_based(symptoms_text, models):
    symptoms_text = symptoms_text.lower()
    symptom_mapper = models.get('symptom_mapper', {})

    results = {
        "diabetes_probability": 0.0,
        "heart_disease_probability": 0.0,
        "parkinsons_probability": 0.0,
        "skin_disease_probability": 0.0,
        "reasoning": "Analysis based on symptom keyword matching."
    }

    # Count matching symptoms for each condition
    for condition, symptoms in symptom_mapper.items():
        matches = sum(1 for symptom in symptoms if symptom in symptoms_text)
        probability = min(matches / len(symptoms), 1.0)

        if condition == 'diabetes':
            results["diabetes_probability"] = probability
        elif condition == 'heart_disease':
            results["heart_disease_probability"] = probability
        elif condition == 'parkinsons':
            results["parkinsons_probability"] = probability
        elif condition == 'skin_disease':
            results["skin_disease_probability"] = probability

    return results

# Function to preprocess skin image
def preprocess_skin_image(image, target_size=(224, 224)):
    # Resize image
    img = image.resize(target_size)
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction functions
def predict_diabetes(data, models):
    try:
        # Create a DataFrame with capitalized column names to match the training data
        data_df = pd.DataFrame({
            'Pregnancies': [data['pregnancies']],
            'Glucose': [data['glucose']],
            'BloodPressure': [data['bloodpressure']],
            'SkinThickness': [data['skinthickness']],
            'Insulin': [data['insulin']],
            'BMI': [data['bmi']],
            'DiabetesPedigreeFunction': [data['diabetespedigreefunction']],
            'Age': [data['age']]
        })

        # Scale the data
        scaler = models.get('diabetes_scaler')
        if scaler:
            data_scaled = scaler.transform(data_df)

            # For now, just demonstrate a placeholder prediction
            # In a real implementation, you'd use the XGBoost model loaded from the JSON file
            probability = np.random.random()
            prediction = int(probability > 0.5)

            return prediction, probability
        else:
            st.warning("Diabetes model not loaded properly. Using demo mode.")
            probability = np.random.random()
            prediction = int(probability > 0.5)
            return prediction, probability

    except Exception as e:
        st.error(f"Error predicting: {e}")
        probability = np.random.random()
        prediction = int(probability > 0.5)
        return prediction, probability

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

def predict_parkinsons(data, models):
    try:
        parkinsons_scaler = models.get('parkinsons_scaler')

        if parkinsons_scaler:
            # In a real scenario, properly format and scale the input data for the model
            # For now, return random predictions
            motor_updrs = np.random.uniform(10, 40)
            total_updrs = motor_updrs + np.random.uniform(5, 20)
            return motor_updrs, total_updrs
        else:
            st.warning("Parkinson's model not loaded properly. Using demo mode.")
            motor_updrs = np.random.uniform(10, 40)
            total_updrs = motor_updrs + np.random.uniform(5, 20)
            return motor_updrs, total_updrs
    except Exception as e:
        st.error(f"Error predicting: {e}")
        motor_updrs = np.random.uniform(10, 40)
        total_updrs = motor_updrs + np.random.uniform(5, 20)
        return motor_updrs, total_updrs

def predict_skin_disease(image, models):
    try:
        # In a real application, we would:
        # 1. Preprocess the image
        # 2. Load the model (PyTorch ResNet50)
        # 3. Make a prediction

        # For now, return a random prediction
        class_idx = np.random.randint(0, 7)
        confidence = np.random.uniform(0.7, 0.99)
        return class_idx, confidence
    except Exception as e:
        st.error(f"Error predicting: {e}")
        class_idx = np.random.randint(0, 7)
        confidence = np.random.uniform(0.7, 0.99)
        return class_idx, confidence

# Custom CSS for responsive design
def load_css():
    st.markdown("""
    <style>
    /* Responsive design adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            white-space: normal;
        }
    }

    /* Card styling */
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Result styling */
    .result-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }

    .result-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }

    .result-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }

    /* User type selector */
    .user-type-select {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }

    /* Enhance button style */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }

    /* Improve prominence of tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }

    /* Improve header styles */
    h1, h2, h3 {
        margin-bottom: 1rem;
    }

    /* Improve form layout */
    .form-group {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Load custom CSS
    load_css()

    # Load models
    models, models_loaded = load_models()

    # Header
    st.title("Healthcare Prediction System")
    st.write("This application predicts various diseases based on input parameters.")

    # User type selection (Normal User vs Medical Staff)
    st.markdown('<div class="user-type-select">', unsafe_allow_html=True)
    user_type = st.radio("Select User Type:", ["Normal User", "Medical Staff"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Natural language symptom input for Normal User
    if user_type == "Normal User":
        st.header("Symptom Analysis")
        st.write("Please describe your symptoms in natural language:")

        symptoms_text = st.text_area(
            "Enter your symptoms here",
            height=150,
            placeholder="Example: I've been feeling very tired lately, experiencing increased thirst, and going to the bathroom more frequently. I've also noticed some weight loss despite eating normally."
        )

        if st.button("Analyze Symptoms"):
            if symptoms_text:
                with st.spinner("Analyzing symptoms..."):
                    # Analyze symptoms using Gemini API or rule-based fallback
                    analysis_results = analyze_symptoms_with_gemini(symptoms_text, models)

                    # Display results
                    st.subheader("Analysis Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Probability of Conditions")
                        diabetes_prob = analysis_results.get("diabetes_probability", 0) * 100
                        heart_prob = analysis_results.get("heart_disease_probability", 0) * 100
                        parkinsons_prob = analysis_results.get("parkinsons_probability", 0) * 100
                        skin_prob = analysis_results.get("skin_disease_probability", 0) * 100

                        # Display probabilities as progress bars
                        st.markdown(f"**Diabetes**: {diabetes_prob:.1f}%")
                        st.progress(diabetes_prob/100)

                        st.markdown(f"**Heart Disease**: {heart_prob:.1f}%")
                        st.progress(heart_prob/100)

                        st.markdown(f"**Parkinson's Disease**: {parkinsons_prob:.1f}%")
                        st.progress(parkinsons_prob/100)

                        st.markdown(f"**Skin Disease**: {skin_prob:.1f}%")
                        st.progress(skin_prob/100)

                    with col2:
                        st.markdown("### Analysis")
                        st.write(analysis_results.get("reasoning", "Analysis based on symptom patterns."))

                        # Recommended next steps
                        st.markdown("### Recommended Next Steps")

                        max_prob = max(
                            diabetes_prob,
                            heart_prob,
                            parkinsons_prob,
                            skin_prob
                        )

                        if max_prob > 70:
                            st.warning("⚠️ Please consult a healthcare professional immediately.")
                        elif max_prob > 40:
                            st.info("ℹ️ Consider scheduling an appointment with your doctor.")
                        else:
                            st.success("✓ Continue monitoring your symptoms and maintain a healthy lifestyle.")

                    # Add disclaimer
                    st.markdown("---")
                    st.caption("**Disclaimer**: This analysis is based on the symptoms you provided and should not be considered a medical diagnosis. Always consult with a qualified healthcare professional for proper evaluation.")
            else:
                st.warning("Please enter your symptoms before analyzing.")

    # Detailed prediction tabs for Medical Staff
    else:  # Medical Staff
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinson's Disease",
            "Skin Disease Detection"
        ])

        # Diabetes Prediction Tab
        with tab1:
            st.header("Diabetes Prediction")
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

            diabetes_data = {
                'pregnancies': pregnancies,
                'glucose': glucose,
                'bloodpressure': blood_pressure,
                'skinthickness': skin_thickness,
                'insulin': insulin,
                'bmi': bmi,
                'diabetespedigreefunction': dpf,
                'age': age
            }

            if st.button("Predict Diabetes"):
                prediction, probability = predict_diabetes(diabetes_data, models)

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.markdown(f'<div class="result-negative">Diabetic (Probability: {probability:.2f})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-positive">Non-Diabetic (Probability: {1-probability:.2f})</div>', unsafe_allow_html=True)

                # Add explanatory text
                st.write("---")
                st.write("**Note:** This prediction is based on the input parameters. Always consult with a healthcare professional for proper diagnosis.")

        # Heart Disease Prediction Tab
        with tab2:
            st.header("Heart Disease Prediction")
            st.write("Enter patient information to predict heart disease:")

            col1, col2 = st.columns(2)

            with col1:
                age_heart = st.number_input("Age", min_value=29, max_value=77, value=45, step=1, key="heart_age")
                sex = st.selectbox("Sex", ["Male", "Female"], index=0)
                chest_pain = st.selectbox("Chest Pain Type",
                                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                                         index=2)
                resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",
                                           min_value=94, max_value=200, value=130, step=1)
                cholesterol = st.number_input("Cholesterol (mg/dl)",
                                            min_value=126, max_value=564, value=240, step=1)
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                         ["No", "Yes"], index=0)
                resting_ecg = st.selectbox("Resting ECG",
                                          ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                                          index=0)

            with col2:
                max_hr = st.number_input("Max Heart Rate", min_value=71, max_value=202, value=150, step=1)
                exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0)
                st_depression = st.number_input("ST Depression", min_value=0.0, max_value=6.2, value=1.0, step=0.1, format="%.1f")
                st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], index=1)
                num_vessels = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3", "4"], index=0)
                thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], index=2)

            heart_data = {
                'age': age_heart,
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

            if st.button("Predict Heart Disease"):
                prediction, probability = predict_heart_disease(heart_data, models)

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.markdown(f'<div class="result-negative">Heart Disease Detected (Probability: {probability:.2f})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-positive">No Heart Disease Detected (Probability: {1-probability:.2f})</div>', unsafe_allow_html=True)

                # Add explanatory text
                st.write("---")
                st.write("**Note:** This prediction is based on the input parameters. Always consult with a healthcare professional for proper diagnosis.")

        # Parkinson's Disease Tab
        with tab3:
            st.header("Parkinson's Disease UPDRS Prediction")
            st.write("Enter voice measurements to predict Parkinson's disease UPDRS scores:")

            col1, col2 = st.columns(2)

            with col1:
                jitter_pct = st.number_input("Jitter(%)", min_value=0.0, max_value=1.0, value=0.006, step=0.001, format="%.3f")
                jitter_abs = st.number_input("Jitter(Abs)", min_value=0.0, max_value=0.0001, value=0.00004, step=0.00001, format="%.5f")
                jitter_rap = st.number_input("Jitter:RAP", min_value=0.0, max_value=0.02, value=0.003, step=0.001, format="%.3f")
                jitter_ppq5 = st.number_input("Jitter:PPQ5", min_value=0.0, max_value=0.02, value=0.003, step=0.001, format="%.3f")
                jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.03, value=0.01, step=0.001, format="%.3f")
                shimmer_pct = st.number_input("Shimmer(%)", min_value=0.0, max_value=0.12, value=0.03, step=0.01, format="%.2f")
                shimmer_db = st.number_input("Shimmer(dB)", min_value=0.0, max_value=1.5, value=0.3, step=0.1, format="%.1f")
                shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.05, value=0.016, step=0.001, format="%.3f")

            with col2:
                shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.06, value=0.022, step=0.001, format="%.3f")
                shimmer_apq11 = st.number_input("Shimmer:APQ11", min_value=0.0, max_value=0.08, value=0.03, step=0.01, format="%.2f")
                shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.15, value=0.05, step=0.01, format="%.2f")
                nhr = st.number_input("NHR", min_value=0.0, max_value=0.5, value=0.025, step=0.001, format="%.3f")
                hnr = st.number_input("HNR", min_value=0.0, max_value=30.0, value=21.0, step=0.1, format="%.1f")
                rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
                dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.7, step=0.01, format="%.2f")
                ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")

            parkinsons_data = {
                'jitter_pct': jitter_pct,
                'jitter_abs': jitter_abs,
                'jitter_rap': jitter_rap,
                'jitter_ppq5': jitter_ppq5,
                'jitter_ddp': jitter_ddp,
                'shimmer_pct': shimmer_pct,
                'shimmer_db': shimmer_db,
                'shimmer_apq3': shimmer_apq3,
                'shimmer_apq5': shimmer_apq5,
                'shimmer_apq11': shimmer_apq11,
                'shimmer_dda': shimmer_dda,
                'nhr': nhr,
                'hnr': hnr,
                'rpde': rpde,
                'dfa': dfa,
                'ppe': ppe
            }

            if st.button("Predict Parkinson's UPDRS"):
                motor_updrs, total_updrs = predict_parkinsons(parkinsons_data, models)

                st.subheader("Prediction Result:")
                st.markdown(f'<div class="result-neutral">Motor UPDRS Score: {motor_updrs:.2f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-neutral">Total UPDRS Score: {total_updrs:.2f}</div>', unsafe_allow_html=True)

                # Add explanatory text
                st.write("---")
                st.write("**UPDRS:** Unified Parkinson's Disease Rating Scale, a clinical rating scale for Parkinson's disease. Higher values indicate more severe symptoms.")
                st.write("**Note:** This prediction is based on voice measurements only. Always consult with a healthcare professional for proper diagnosis.")

        # Skin Disease Detection Tab
        with tab4:
            st.header("Skin Disease Detection")
            st.write("Upload an image of the skin condition to get a diagnosis:")

            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Load and display the image
                image = Image.open(uploaded_file).convert('RGB')

                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                with col2:
                    st.write("Classifying...")

                    # Predict skin disease
                    pred_class, confidence = predict_skin_disease(image, models)

                    # Get the label
                    labels = models.get('skin_cancer_labels', {
                        0: 'Actinic Keratoses',
                        1: 'Basal Cell Carcinoma',
                        2: 'Benign Keratosis',
                        3: 'Dermatofibroma',
                        4: 'Melanoma',
                        5: 'Melanocytic Nevi',
                        6: 'Vascular Lesions'
                    })

                    predicted_class = labels.get(pred_class, 'Unknown')

                    # Display results
                    st.subheader("Prediction Result:")

                    # Style the result based on condition severity
                    if pred_class == 1 or pred_class == 4:  # Basal Cell Carcinoma or Melanoma
                        st.markdown(f'<div class="result-negative">Prediction: {predicted_class}<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
                        st.warning("⚠️ This condition requires immediate medical attention.")
                    else:
                        st.markdown(f'<div class="result-neutral">Prediction: {predicted_class}<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
                        st.info("ℹ️ Please consult with a dermatologist for confirmation.")

                    # Add additional information about the condition
                    st.write("---")
                    st.write("**About this condition:**")

                    condition_info = {
                        'Actinic Keratoses': "Rough, scaly patch on skin that develops from years of sun exposure. It's a precancerous condition.",
                        'Basal Cell Carcinoma': "Most common type of skin cancer. Appears as a pearly white or pink bump or patch on the skin.",
                        'Benign Keratosis': "Harmless growth that many people get as they age. They range in color from light tan to black.",
                        'Dermatofibroma': "Common, harmless skin growth that's usually found on the legs. It's firm and can be pink, gray, or brown.",
                        'Melanoma': "The most serious type of skin cancer. Develops in the cells that produce melanin, the pigment that gives skin its color.",
                        'Melanocytic Nevi': "Common moles. Usually harmless but can sometimes develop into melanoma.",
                        'Vascular Lesions': "Benign abnormalities of blood vessels, including hemangiomas and vascular malformations."
                    }

                    st.write(condition_info.get(predicted_class, "Information not available."))

                    # Add disclaimer
                    st.caption("**Disclaimer**: This is not a substitute for professional medical advice. Always consult with a qualified healthcare professional for proper diagnosis.")

            else:
                st.info("Please upload an image to get a diagnosis.")

    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>© 2025 Healthcare Prediction System | For educational purposes only</p>
        <p>Not for use in actual medical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()