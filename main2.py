import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load the classification model using pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="Aimlab/xlm-roberta-roman-urdu-finetuned")

# Load model directly
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Aimlab/xlm-roberta-roman-urdu-finetuned")
    model = AutoModelForSequenceClassification.from_pretrained("Aimlab/xlm-roberta-roman-urdu-finetuned")
    return tokenizer, model

# Initialize the Streamlit app
def main():
    st.title("Text Classification in Roman Urdu")
    st.write("Enter text below and get the classification.")

    # Load the pipeline and model
    classification_pipeline = load_pipeline()
    tokenizer, model = load_model()

    # Input text box for user
    text_input = st.text_area("Input Text", height=150)

    if st.button("Classify Text"):
        if text_input:
            # Use the pipeline for classification
            results = classification_pipeline(text_input)

            st.subheader("Classification Results:")
            for result in results:
                label = result['label']
                score = result['score']
                st.write(f"**Label:** {label} | **Score:** {score:.4f}")
        else:
            st.error("Please enter some text to classify.")

if __name__ == "__main__":
    main()
