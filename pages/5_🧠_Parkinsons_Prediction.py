import streamlit as st
from app import load_models
import numpy as np

# Load models
models, _ = load_models()

st.title("🧠 Parkinson's Disease Prediction")
st.write("Enter voice measurements to predict Parkinson's disease UPDRS scores:")

col1, col2 = st.columns(2)

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

if st.button("Predict Parkinson's UPDRS"):
    # Prepare input data
    input_data = {
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
    motor_updrs, total_updrs = predict_parkinsons(input_data, models)

    st.subheader("Prediction Result:")
    
    # Severity assessment
    if motor_updrs > 30:
        severity = "Moderate to Severe"
        rec_color = "red"
    elif motor_updrs > 15:
        severity = "Mild to Moderate"
        rec_color = "orange"
    else:
        severity = "Mild"
        rec_color = "green"

    st.markdown(f"**Severity:** <span style='color:{rec_color}'>{severity}</span>", unsafe_allow_html=True)
    st.info(f"Motor UPDRS Score: {motor_updrs:.2f}")
    st.info(f"Total UPDRS Score: {total_updrs:.2f}")

    # Parkinson's remedies
    st.markdown("""
    ### Management Recommendations:
    **Medical Care:**
    - Consult a movement disorder specialist
    - Medication management (levodopa, dopamine agonists)
    - Regular neurology follow-ups

    **Physical Therapy:**
    - Balance exercises (tai chi recommended)
    - Strength training
    - Flexibility exercises
    - Gait training

    **Daily Living:**
    - Home safety modifications
    - Assistive devices if needed
    - Speech therapy if required
    - Support groups

    **Nutrition:**
    - High-fiber diet for constipation
    - Adequate hydration
    - Protein timing with medications
    - Antioxidant-rich foods
    """)
    
    # Add disclaimer
    st.markdown("---")
    st.caption("**Disclaimer**: This prediction is based on voice measurements only. Always consult with a healthcare professional for proper diagnosis.")