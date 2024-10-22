import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from typing import Optional
import time

class EnglishToRomanUrduTranslator:
    def __init__(self, model_name: str = "facebook/bart-large"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise

    def translate(self, text: str, max_length: int = 128) -> Optional[str]:
        """Translate English text to Roman Urdu"""
        try:
            # Prepare the input text
            inputs = self.tokenizer.encode(
                f"translate English to Roman Urdu: {text}",
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(self.device)

            # Generate translation
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            # Decode the translation
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text

        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return None

class TranslatorUI:
    def __init__(self):
        self.setup_page()
        self.translator = self.load_translator()

    @staticmethod
    def setup_page():
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="English to Roman Urdu Translator (BART)",
            page_icon="🔄",
            layout="wide"
        )
        
        st.markdown("""
            <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .translation-box {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9f9f9;
            }
            .stButton>button {
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_resource
    def load_translator():
        """Load and cache the translator"""
        return EnglishToRomanUrduTranslator()

    def create_sidebar(self):
        """Create the sidebar with translation settings"""
        with st.sidebar:
            st.header("Translation Settings")
            max_length = st.slider(
                "Maximum Length",
                min_value=50,
                max_value=250,
                value=128,
                help="Maximum length of the translated text"
            )
            return max_length

    def run(self):
        """Run the Streamlit application"""
        st.title("🔄 English to Roman Urdu Translator")
        st.markdown("#### Using BART Neural Machine Translation")
        
        max_length = self.create_sidebar()

        # Create two columns for input and output
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("English Input")
            input_text = st.text_area(
                "",
                height=200,
                placeholder="Enter your English text here...",
                key="input"
            )

        with col2:
            st.subheader("Roman Urdu Output")
            output_placeholder = st.empty()

        # Add translation button with loading state
        if st.button("🔄 Translate Text", use_container_width=True):
            if input_text:
                with st.spinner("Translating..."):
                    # Add artificial delay to show processing
                    time.sleep(0.5)
                    translation = self.translator.translate(input_text, max_length)
                    
                    if translation:
                        output_placeholder.markdown(
                            f'<div class="translation-box">{translation}</div>',
                            unsafe_allow_html=True
                        )
                        st.success("Translation completed! ✨")
            else:
                st.warning("⚠️ Please enter some text to translate")

        # Add information section
        st.markdown("---")
        st.markdown("""
        ### About this Translator
        
        This translator uses the BART (Bidirectional and Auto-Regressive Transformers) model with the following features:
        
        - **Advanced Beam Search**: Uses 5 beams for better translation quality
        - **Temperature Control**: Set to 0.7 for balanced creativity and accuracy
        - **Top-k & Top-p Sampling**: Ensures diverse and fluent translations
        - **Length Penalty**: Optimized for maintaining appropriate translation length
        
        ⚠️ Note: This model requires fine-tuning for optimal Roman Urdu translations. Results may vary.
        """)

if __name__ == "__main__":
    app = TranslatorUI()
    app.run()
