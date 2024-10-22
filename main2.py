import streamlit as st
from transformers import pipeline

# Load the translation model using pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("translation", model="HaiderSultanArc/t5-small-english-to-urdu")

# Initialize the Streamlit app
def main():
    st.title("English to Roman Urdu Translator")
    st.write("Enter English text below and get the translation in Urdu.")

    # Load the pipeline
    translation_pipeline = load_pipeline()

    # Input text box for user
    english_text = st.text_area("Input English Text", height=150)

    if st.button("Translate to Urdu"):
        if english_text:
            # Use the pipeline for translation
            translated_text = translation_pipeline(english_text, max_length=400)[0]['translation_text']

            st.subheader("Translated Urdu Text:")
            st.write(translated_text)
            # Here you can implement a way to convert Urdu to Roman Urdu if necessary.
        else:
            st.error("Please enter some text to translate.")

if __name__ == "__main__":
    main()
