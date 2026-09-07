+import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect, DetectorFactory
import warnings
from openai import OpenAI

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

class MultilingualChatbot:
    """
    A multilingual chatbot with sentiment analysis, context retention,
    and cross-lingual reasoning.
    """
    def __init__(self, translation_model="facebook/nllb-200-distilled-600M", sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        print(f"Initializing Multilingual Chatbot...")
        
        # 1. Load Translation Model (NLLB)
        print(f"Loading translation model: {translation_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(translation_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(translation_model)
        
        # 2. Load Sentiment Analysis Model
        print(f"Loading sentiment analysis model: {sentiment_model}")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model)
        
        # 3. Initialize OpenAI client for reasoning engine
        print("Initializing OpenAI client for reasoning engine (gemini-3-flash-preview)...")
        self.openai_client = OpenAI()
        self.reasoning_model = "gemini-3-flash-preview"
        
        # Configuration
        self.internal_lang = "eng_Latn"
        self.lang_map = {
            'en': 'eng_Latn', 'es': 'spa_Latn', 'fr': 'fra_Latn',
            'de': 'deu_Latn', 'it': 'ita_Latn', 'pt': 'por_Latn',
            'nl': 'nld_Latn', 'zh-cn': 'zho_Hans',
            'ja': 'jpn_Jpan', 'ko': 'kor_Kore', 'ru': 'rus_Cyrl',
            'ar': 'ary_Arab'
        }
        self.history = []
        
        print("Chatbot is ready!")

    def _get_nllb_code(self, lang_code):
        return self.lang_map.get(lang_code, self.internal_lang)

    def detect_language(self, text):
        try:
            return detect(text)
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

    def analyze_sentiment(self, text_en):
        """
        Analyze sentiment of English text.
        Returns: 'positive', 'negative', or 'neutral'
        """
        result = self.sentiment_analyzer(text_en)[0]
        label = result['label'].lower()
        return label

    def process_input(self, user_input):
        # 1. Identify Language
        lang_code = self.detect_language(user_input)
        
        # 2. Translate to English for Sentiment and Reasoning
        input_en = self.translate(user_input, lang_code, 'en')
        
        # 3. Sentiment Analysis
        sentiment = self.analyze_sentiment(input_en)
        
        # 4. Update History
        self.history.append({"role": "user", "content": input_en, "sentiment": sentiment})
        
        # 5. Sentiment-Aware Prompting for OpenAI model
        sentiment_instructions = {
            "positive": "The user is expressing positive emotions. Respond enthusiastically and affirmingly. Be helpful and friendly. Keep the response concise and to the point, but still engaging and complete. Ensure the response is a full, natural sentence or paragraph.",
            "negative": "The user is expressing negative emotions or frustration. Respond empathetically, acknowledge their feelings, and offer concrete help or solutions. Be apologetic if appropriate. Focus on resolving the issue and maintaining a supportive, calm, and reassuring tone. Provide a complete, helpful, and natural response, avoiding abrupt endings.",
            "neutral": "The user's sentiment is neutral. Respond professionally, clearly, and concisely. Provide direct answers or relevant information without excessive emotional expression. Be helpful and friendly. Ensure the response is a complete, natural sentence or paragraph."
        }
        
        system_message_content = sentiment_instructions.get(sentiment, "Be a helpful, friendly, and empathetic AI assistant. Provide complete and helpful responses.")
        
        messages = [
            {"role": "system", "content": system_message_content}
        ]
        
        # Add recent conversation history to the messages list for context
        # Ensure the history is alternating user/assistant messages
        for turn in self.history[-4:]:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": turn['content']})
            elif "response" in turn:
                messages.append({"role": "assistant", "content": turn['response']})
        
        # Ensure the last message is the current user input
        if messages[-1]["role"] != "user" or messages[-1]["content"] != input_en:
            messages.append({"role": "user", "content": input_en})

        # 6. Generate Response using OpenAI
        assistant_response_en = "I'm having trouble generating a response right now. Please try again."
        try:
            response = self.openai_client.chat.completions.create(
                model=self.reasoning_model,
                messages=messages,
                max_tokens=350, # Increased max tokens for more elaborate responses
                temperature=0.8 # Adjusted temperature slightly for more varied but still coherent responses
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                assistant_response_en = response.choices[0].message.content.strip()
            else:
                print(f"OpenAI response was empty or malformed: {response}")
        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")

        # 7. Translate back to user's language
        final_response = self.translate(assistant_response_en, 'en', lang_code)
        
        # Store in history
        self.history[-1]["response"] = assistant_response_en
        
        return {
            "detected_language": lang_code,
            "sentiment": sentiment,
            "internal_response": assistant_response_en,
            "final_output": final_response
        }

if __name__ == "__main__":
    bot = MultilingualChatbot()
    
    print("\n" + "="*50)
    print("   SENTIMENT-AWARE MULTILINGUAL CHATBOT (v8 - Refined Prompting & Params)")
    print("="*50 + "\n")
    
    test_inputs = [
        "I am so happy with your service! It's amazing.", # Positive
        "This is the worst experience I've ever had. I'm very angry.", # Negative
        "Can you tell me the weather in London?", # Neutral
        "Je suis très content de votre aide.", # Positive (French)
        "No me gusta nada como funciona esto.", # Negative (Spanish)
        "That's fantastic! I really appreciate your quick response.", # Positive follow-up
        "I'm still having issues, this is not resolved.", # Negative follow-up
        "What else can you do?" # Neutral follow-up
    ]
    
    for inp in test_inputs:
        print(f"You: {inp}")
        res = bot.process_input(inp)
        print(f"[Sentiment: {res['sentiment'].upper()}]")
        print(f"Chatbot: {res['final_output']}\n")
        print("-" * 50)
