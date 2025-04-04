import os
import streamlit as st
import joblib
import json
import pickle
# Suppress warnings
import warnings
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
        with open('models/diabetes_rf_model.pkl', 'rb') as f:
            models['diabetes_model'] = pickle.load(f)

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

# Main app
def main():
    # Load models
    models, models_loaded = load_models()

    # Store models in session state
    if 'models' not in st.session_state:
        st.session_state.models = models

    # Home page content
    st.title("🏥 Healthcare Prediction System")
    st.write("Welcome to the Healthcare Prediction System! Use the sidebar to navigate through different prediction tools.")

    st.markdown("""
    ### Features:
    - **Symptom Analysis**: Describe your symptoms and get a preliminary analysis.
    - **Diabetes Prediction**: Predict the likelihood of diabetes based on health metrics.
    - **Heart Disease Prediction**: Assess the risk of heart disease.
    - **Parkinson's Prediction**: Predict Parkinson's disease progression using voice measurements.
    - **Skin Disease Detection**: Upload an image of a skin condition for diagnosis.
    """)

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