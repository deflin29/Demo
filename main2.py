import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def load_model():
    # Load the model and tokenizer
    model_name = "t5-base"  # You can also use "t5-small" for faster but potentially less accurate results
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    # Prepare the input text
    input_text = f"translate English to Roman Urdu: {text}"
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    # Decode the translation
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def main():
    st.set_page_config(page_title="English to Roman Urdu Translator", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 800px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("✨ English to Roman Urdu Translator")
    st.markdown("---")
    
    # Load model and tokenizer
    @st.cache_resource
    def get_model():
        return load_model()
    
    tokenizer, model = get_model()
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("English Input")
        input_text = st.text_area("", height=200, placeholder="Enter English text here...")
        
    with col2:
        st.subheader("Roman Urdu Output")
        output_placeholder = st.empty()
    
    # Add translation button
    if st.button("🔄 Translate", use_container_width=True):
        if input_text:
            with st.spinner("Translating..."):
                try:
                    translation = translate_text(input_text, tokenizer, model)
                    output_placeholder.text_area("", translation, height=200, disabled=True)
                    st.success("Translation completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during translation: {str(e)}")
        else:
            st.warning("Please enter some text to translate")
    
    # Add information about the model
    st.markdown("---")
    st.markdown("""
        **About this translator:**
        - Uses T5 model for translation
        - Supports multiple sentences
        - Maximum input length: 512 tokens
        - Uses beam search for better translation quality
    """)

if __name__ == "__main__":
    main()
