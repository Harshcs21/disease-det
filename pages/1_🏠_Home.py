import streamlit as st

st.title("🏠 Home")
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