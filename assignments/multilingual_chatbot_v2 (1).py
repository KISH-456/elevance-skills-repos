import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect, DetectorFactory
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

class MultilingualChatbot:
    """
    A multilingual chatbot that supports automatic language identification,
    context retention across language switches, and cross-lingual reasoning.
    """
    def __init__(self, translation_model="facebook/nllb-200-distilled-600M"):
        print(f"Initializing Multilingual Chatbot with {translation_model}...")
        
        # Load NLLB model for translation (supports 200 languages)
        self.tokenizer = AutoTokenizer.from_pretrained(translation_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(translation_model)
        
        # Internal processing language
        self.internal_lang = "eng_Latn" 
        
        # Mapping for langdetect codes to NLLB codes
        self.lang_map = {
            'en': 'eng_Latn',
            'es': 'spa_Latn',
            'fr': 'fra_Latn',
            'de': 'deu_Latn',
            'it': 'ita_Latn',
            'pt': 'por_Latn',
            'nl': 'nld_Latn',
            'zh-cn': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'ko': 'kor_Kore',
            'ru': 'rus_Cyrl',
            'ar': 'ary_Arab'
        }
        
        # Conversational memory
        self.history = []
        
        # Using a slightly better model for reasoning if possible, but keeping it open-source and lightweight
        # In a real scenario, use Llama-3-8B or similar. Here we use GPT-2 with better prompting.
        print("Loading reasoning engine (GPT-2)...")
        self.reasoning_engine = pipeline("text-generation", model="gpt2")
        
        print("Chatbot is ready!")

    def _get_nllb_code(self, lang_code):
        return self.lang_map.get(lang_code, self.internal_lang)

    def detect_language(self, text):
        try:
            detected = detect(text)
            return detected
        except:
            return 'en'

    def translate(self, text, source_lang_code, target_lang_code):
        if source_lang_code == target_lang_code:
            return text
            
        source_nllb = self._get_nllb_code(source_lang_code)
        target_nllb = self._get_nllb_code(target_lang_code)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_nllb),
            max_length=100
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def process_input(self, user_input):
        # 1. Identify Language
        lang_code = self.detect_language(user_input)
        
        # 2. Translate to Internal Language (English) for reasoning
        input_en = self.translate(user_input, lang_code, 'en')
        
        # 3. Maintain Context and Continuity
        # Add current turn to history
        self.history.append({"role": "user", "content": input_en, "lang": lang_code})
        
        # Construct prompt with history
        # We use a structured prompt to help GPT-2 stay on track
        prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\n"
        for turn in self.history[-3:]: # Keep last 3 turns for context
            prompt += f"User: {turn['content']}\n"
            if "response" in turn:
                prompt += f"Assistant: {turn['response']}\n"
        prompt += "Assistant:"
        
        # 4. Generate Response (Reasoning)
        output = self.reasoning_engine(
            prompt, 
            max_new_tokens=40, 
            pad_token_id=50256,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            truncation=True
        )[0]['generated_text']
        
        # Extract assistant response
        try:
            assistant_response_en = output.split("Assistant:")[-1].strip().split("\n")[0]
        except:
            assistant_response_en = "I'm here to help!"

        # 5. Translate back to user's language
        final_response = self.translate(assistant_response_en, 'en', lang_code)
        
        # Store response in history
        self.history[-1]["response"] = assistant_response_en
        
        return {
            "detected_language": lang_code,
            "translated_input": input_en,
            "internal_response": assistant_response_en,
            "final_output": final_response
        }

if __name__ == "__main__":
    bot = MultilingualChatbot()
    
    print("\n" + "="*50)
    print("      MULTILINGUAL CHATBOT DEMO (v2)")
    print("="*50 + "\n")
    
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye! ¡Adiós! Au revoir! Tschüss!")
            break
            
        res = bot.process_input(user_msg)
        print(f"\n[System Info]")
        print(f"  Detected Language: {res['detected_language']}")
        print(f"  Reasoning (EN):    {res['internal_response']}")
        print(f"\nChatbot: {res['final_output']}\n")
        print("-" * 50)
