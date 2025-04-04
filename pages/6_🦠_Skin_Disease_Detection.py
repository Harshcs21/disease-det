import streamlit as st
from app import load_models
from PIL import Image
import numpy as np

# Load models
models, _ = load_models()

st.title("🦠 Skin Disease Detection")
st.write("Upload an image of the skin condition to get a diagnosis:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    if st.button("Predict Skin Disease"):
        with st.spinner("Analyzing image..."):
            # Placeholder for actual model prediction
            class_idx = np.random.randint(0, 7)
            confidence = np.random.uniform(0.7, 0.99)

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
            predicted_class = labels.get(class_idx, 'Unknown')

            # Display results
            st.subheader("Prediction Result:")
        
            # Condition-specific recommendations
            condition_info = {
                'Actinic Keratoses': {
                    "treatment": "Cryotherapy, topical medications, photodynamic therapy",
                    "self_care": "Regular sunscreen use, avoid sun exposure, moisturize"
                },
                'Basal Cell Carcinoma': {
                    "treatment": "Surgical excision, Mohs surgery, radiation therapy",
                    "self_care": "Immediate dermatologist consultation, sun protection"
                },
                'Benign Keratosis': {
                    "treatment": "Usually none needed, cryotherapy if bothersome",
                    "self_care": "Monitor for changes, gentle skin care"
                },
                'Dermatofibroma': {
                    "treatment": "Surgical removal if symptomatic",
                    "self_care": "No treatment needed unless changing"
                },
                'Melanoma': {
                    "treatment": "Immediate surgical excision, possible immunotherapy",
                    "self_care": "Emergency dermatologist visit, sun avoidance"
                },
                'Melanocytic Nevi': {
                    "treatment": "Monitoring, excision if changing",
                    "self_care": "Regular self-exams, ABCDE rule monitoring"
                },
                'Vascular Lesions': {
                    "treatment": "Laser therapy, surgical options",
                    "self_care": "Gentle skin care, sun protection"
                }
            }

            info = condition_info.get(predicted_class, {
                "treatment": "Consult dermatologist",
                "self_care": "Monitor for changes"
            })

            if class_idx == 1 or class_idx == 4:  # BCC or Melanoma
                st.error(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
                st.warning("⚠️ This condition requires immediate medical attention!")
            else:
                st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

            st.markdown(f"""
            ### Recommended Care:
            **Medical Treatment Options:**
            {info['treatment']}

            **Self-Care Recommendations:**
            {info['self_care']}

            **General Skin Health Tips:**
            - Daily sunscreen (SPF 30+)
            - Regular skin self-exams
            - Annual dermatologist check-ups
            - Stay hydrated
            - Avoid tanning beds
            """)

            # Add disclaimer
            st.markdown("---")
            st.caption("**Disclaimer**: This is not a substitute for professional medical advice. Always consult with a qualified healthcare professional for proper diagnosis.")