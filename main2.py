import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

def load_model():
    # Load the model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-ur"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translation
    outputs = model.generate(**inputs)
    
    # Decode the translation
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def main():
    st.title("English to Roman Urdu Translator")
    st.write("Enter English text below to translate it to Roman Urdu")
    
    # Load model and tokenizer
    @st.cache_resource
    def get_model():
        return load_model()
    
    tokenizer, model = get_model()
    
    # Create input text area
    input_text = st.text_area("Enter English Text:", height=150)
    
    if st.button("Translate"):
        if input_text:
            with st.spinner("Translating..."):
                translation = translate_text(input_text, tokenizer, model)
                st.success("Translation Complete!")
                st.write("Roman Urdu Translation:")
                st.write(translation)
        else:
            st.warning("Please enter some text to translate")

if __name__ == "__main__":
    main()
