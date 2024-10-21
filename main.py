

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer from Hugging Face
model_name = "your_model_name_here"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_to_roman_urdu(text):
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    # Perform translation
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    # Decode the output to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Streamlit app UI
st.title("English to Roman Urdu Translator")
st.write("Enter an English text below to get its translation in Roman Urdu.")

# Input text
input_text = st.text_area("English Text", "")

if st.button("Translate"):
    if input_text:
        # Translate the input text
        roman_urdu_translation = translate_to_roman_urdu(input_text)
        # Display the translation
        st.write("**Roman Urdu Translation:**")
        st.write(roman_urdu_translation)
    else:
        st.warning("Please enter some text for translation.")
