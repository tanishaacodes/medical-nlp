import streamlit as st
from preprocess import get_word_segments  # Changed from clean_handwriting
from ocr_engine import MedReader
from medical_nlp import autocorrect_medical, translate_medical_shorthand # Added shorthand

st.set_page_config(page_title="MedScript OCR", page_icon="👨‍⚕️")
st.title("Medical Handwriting Detector")
st.write("Upload a handwritten prescription")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Display Original
    st.image(uploaded_file, caption="Original Prescription", use_container_width=True)
    
    # Initialize Model (One time)
    @st.cache_resource
    def load_model():
        return MedReader()
    
    reader = load_model()

    with st.spinner('Segmenting and reading words...'):
        # 2. Get individual word images from the handwriting
        # We use the new segmentation function from Step 1
        word_images = get_word_segments(uploaded_file)
        
        full_raw_text = []
        full_corrected_text = []
        total_confidence = 0

        # 3. Process each word one by one
        if not word_images:
            st.error("No handwriting detected. Please try a clearer image.")
        else:
            # Create progress bar
            progress_bar = st.progress(0)
            
            for i, word_pil in enumerate(word_images):
                # Predict
                raw_word, confidence = reader.predict(word_pil)
                
                # Autocorrect (Amoxioillin -> Amoxicillin)
                corrected_word = autocorrect_medical(raw_word)
                
                # Translate Shorthand (bid -> twice a day)
                final_word = translate_medical_shorthand(corrected_word)
                
                # Store results
                full_raw_text.append(raw_word)
                full_corrected_text.append(final_word)
                total_confidence += confidence
                
                # Update progress
                progress_bar.progress((i + 1) / len(word_images))

            # Calculate Average Confidence
            avg_confidence = total_confidence / len(word_images)

            # 4. Results Display
            st.divider()
            st.subheader("📝 Digital Transcription")
            
            # Combine the words into a single sentence
            final_output = " ".join(full_corrected_text)
            st.success(f"**Result:** {final_output}")

            # 5. Metrics
            col1, col2 = st.columns(2)
            col1.metric("Raw Model Output", " ".join(full_raw_text))
            col2.metric("Avg. Confidence", f"{round(avg_confidence * 100, 1)}%")
            
            # 6. Safety Warnings
            if avg_confidence < 0.70:
                st.warning("⚠️ **Low confidence detected.** Please verify dosages and drug names manually.")
            else:
                st.info("💡 **Tip:** This model is optimized for common medical terms and shorthand.")


