# import streamlit as st
# from transformers import MBartForConditionalGeneration, MBartTokenizer
# import torch
# from typing import Optional
# import time

# class EnglishToRomanUrduTranslator:
#     def __init__(self):
#         self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
#         self.device = "cpu"  # Simplified device handling
#         self.load_model()

#     def load_model(self):
#         """Load the model and tokenizer"""
#         try:
#             self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
#             self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
#             # Set source and target language
#             self.tokenizer.src_lang = "en_XX"
#             self.tokenizer.tgt_lang = "ur_IN"  # Changed to ur_IN which is supported by mBART-50
#         except Exception as e:
#             st.error(f"Error loading model: {str(e)}")
#             raise

#     def translate(self, text: str, max_length: int = 128) -> Optional[str]:
#         """Translate English text to Roman Urdu"""
#         try:
#             # Prepare the input text with correct language tokens
#             encoded = self.tokenizer(text, 
#                                    return_tensors="pt", 
#                                    padding=True, 
#                                    truncation=True, 
#                                    max_length=max_length)
            
#             # Generate translation
#             generated_tokens = self.model.generate(
#                 encoded["input_ids"],
#                 attention_mask=encoded["attention_mask"],
#                 max_length=max_length,
#                 num_beams=4,
#                 length_penalty=1.0,
#                 early_stopping=True,
#                 forced_bos_token_id=self.tokenizer.lang_code_to_id["ur_IN"]  # Updated language code
#             )

#             # Decode the translation
#             translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
#             return translated_text

#         except Exception as e:
#             st.error(f"Translation error: {str(e)}")
#             return None

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="English to Roman Urdu Translator",
#         page_icon="🔄",
#         layout="wide"
#     )

#     # Custom CSS
#     st.markdown("""
#         <style>
#         .stApp {
#             max-width: 1200px;
#             margin: 0 auto;
#         }
#         .translation-box {
#             border: 1px solid #ddd;
#             border-radius: 5px;
#             padding: 10px;
#             background-color: #f9f9f9;
#         }
#         .stButton>button {
#             width: 100%;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     st.title("🔄 English to Roman Urdu Translator")
#     st.markdown("#### Using Neural Machine Translation")

#     # Initialize translator
#     @st.cache_resource
#     def get_translator():
#         return EnglishToRomanUrduTranslator()

#     translator = get_translator()

#     # Sidebar
#     with st.sidebar:
#         st.header("Translation Settings")
#         max_length = st.slider(
#             label="Maximum Length",
#             min_value=50,
#             max_value=250,
#             value=128,
#             help="Maximum length of the translated text"
#         )

#     # Main content
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("English Input")
#         input_text = st.text_area(
#             label="Enter English text",
#             height=200,
#             placeholder="Enter your English text here...",
#             key="input",
#             label_visibility="collapsed"
#         )

#     with col2:
#         st.subheader("Roman Urdu Output")
#         output_placeholder = st.empty()

#     # Translation button
#     if st.button("🔄 Translate Text", use_container_width=True, key="translate_button"):
#         if input_text:
#             with st.spinner("Translating..."):
#                 translation = translator.translate(input_text, max_length)
                
#                 if translation:
#                     output_placeholder.markdown(
#                         f'<div class="translation-box">{translation}</div>',
#                         unsafe_allow_html=True
#                     )
#                     st.success("Translation completed! ✨")
#         else:
#             st.warning("⚠️ Please enter some text to translate")

#     # Information section
#     st.markdown("---")
#     st.markdown("""
#     ### About this Translator
    
#     This translator uses the mBART-50 Many-to-Many Multilingual Machine Translation model with:
    
#     - **Beam Search**: Uses 4 beams for optimal translation quality
#     - **Length Control**: Adjustable maximum length via slider
#     - **Language Support**: Configured for English to Urdu translation using mBART-50's supported language codes
#     - **Error Handling**: Robust error management and user feedback
    
#     ⚠️ Note: For best results, use clear and concise English text.
#     """)

# if __name__ == "__main__":
#     main()





import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import Optional

class EnglishToRomanUrduTranslator:
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-en-ur"
        self.device = "cpu"
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise

    def translate(self, text: str, max_length: int = 128) -> Optional[str]:
        """Translate English text to Roman Urdu"""
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            
            # Generate translation
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # Decode the translation
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text

        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="English to Roman Urdu Translator",
        page_icon="🌐",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1000px;
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

    st.title("🌐 English to Roman Urdu Translator")
    st.markdown("#### Powered by Helsinki-NLP Translation Model")

    # Initialize translator
    @st.cache_resource
    def get_translator():
        return EnglishToRomanUrduTranslator()

    try:
        translator = get_translator()

        # Main content
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("English Input")
            input_text = st.text_area(
                label="Enter English text",
                height=200,
                placeholder="Enter your English text here...",
                key="input",
                label_visibility="collapsed"
            )

        with col2:
            st.subheader("Roman Urdu Output")
            output_placeholder = st.empty()

        # Translation button
        if st.button("Translate to Urdu", use_container_width=True):
            if input_text:
                with st.spinner("Translating..."):
                    translation = translator.translate(input_text)
                    
                    if translation:
                        output_placeholder.markdown(
                            f'<div class="translation-box">{translation}</div>',
                            unsafe_allow_html=True
                        )
                        st.success("Translation completed! ✨")
            else:
                st.warning("⚠️ Please enter some text to translate")

        # Information section
        st.markdown("---")
        st.markdown("""
        ### About this Translator
        
        This translator uses the Helsinki-NLP model specifically trained for English to Urdu translation:
        
        - Specialized model for English-Urdu translation
        - Optimized for accuracy and performance
        - Supports both short and long text translations
        - Real-time translation with error handling
        
        For best results:
        - Use clear and simple English sentences
        - Avoid using slang or highly technical terms
        - Keep sentences concise for better accuracy
        """)

    except Exception as e:
        st.error(f"Error initializing the translator: {str(e)}")
        st.info("Please try refreshing the page or contact support if the error persists.")

if __name__ == "__main__":
    main()
