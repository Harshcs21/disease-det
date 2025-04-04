import streamlit as st
from app import load_models
import google.generativeai as genai
import json
# Load models
models, _ = load_models()

st.title("🩺 Symptom Analysis")
st.write("Describe your symptoms in natural language to get a preliminary analysis.")

symptoms_text = st.text_area(
    "Enter your symptoms here",
    height=150,
    placeholder="Example: I've been feeling very tired lately, experiencing increased thirst, and going to the bathroom more frequently. I've also noticed some weight loss despite eating normally."
)

def setup_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error setting up Gemini API: {e}")
        return False

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
            with col2:
                st.markdown("### Recommended Actions")
                
                # Diabetes remedies
                if diabetes_prob > 40:
                    st.markdown("""
                    **For Diabetes Prevention/Management:**
                    - Monitor blood sugar regularly
                    - Follow a low-glycemic diet (whole grains, vegetables, lean proteins)
                    - Exercise regularly (30 mins/day)
                    - Maintain healthy weight
                    - Limit processed sugars and carbohydrates
                    """)
                
                # Heart disease remedies
                if heart_prob > 40:
                    st.markdown("""
                    **For Heart Health:**
                    - Adopt a heart-healthy diet (Mediterranean diet recommended)
                    - Exercise regularly (150 mins/week moderate activity)
                    - Manage stress through meditation/yoga
                    - Monitor blood pressure and cholesterol
                    - Quit smoking if applicable
                    """)
                
                # Parkinson's remedies
                if parkinsons_prob > 40:
                    st.markdown("""
                    **For Parkinson's Support:**
                    - Regular physical therapy
                    - Balance exercises (tai chi recommended)
                    - Speech therapy if needed
                    - Medication management with neurologist
                    - Support groups for emotional health
                    """)
                
                # Skin disease remedies
                if skin_prob > 40:
                    st.markdown("""
                    **For Skin Health:**
                    - Use sunscreen daily (SPF 30+)
                    - Avoid excessive sun exposure
                    - Keep skin moisturized
                    - See dermatologist for suspicious moles/changes
                    - Don't pick at skin lesions
                    """)
                
                # Recommended next steps
                st.markdown("### Recommended Next Steps")
                max_prob = max(diabetes_prob, heart_prob, parkinsons_prob, skin_prob)
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