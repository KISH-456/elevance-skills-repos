import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect, DetectorFactory
import warnings
from openai import OpenAI
from knowledge_base_manager import KnowledgeBaseManager

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

class MultilingualChatbot:
    """
    A multilingual chatbot with sentiment analysis, context retention,
    cross-lingual reasoning, and dynamic knowledge base expansion (RAG).
    """
    def __init__(self, translation_model="facebook/nllb-200-distilled-600M", sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        print(f"Initializing Multilingual Chatbot...")
        
        # 1. Load Translation Model (NLLB)
        self.tokenizer = AutoTokenizer.from_pretrained(translation_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(translation_model)
        
        # 2. Load Sentiment Analysis Model
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model)
        
        # 3. Initialize OpenAI client for reasoning engine
        self.openai_client = OpenAI()
        self.reasoning_model = "gemini-3-flash-preview"
        
        # 4. Initialize Knowledge Base Manager (RAG)
        self.kb_manager = KnowledgeBaseManager()
        
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
        result = self.sentiment_analyzer(text_en)[0]
        return result['label'].lower()

    def process_input(self, user_input):
        # 1. Identify Language
        lang_code = self.detect_language(user_input)
        
        # 2. Translate to English for Sentiment and Reasoning
        input_en = self.translate(user_input, lang_code, 'en')
        
        # 3. Sentiment Analysis
        sentiment = self.analyze_sentiment(input_en)
        
        # 4. Knowledge Retrieval (RAG)
        retrieved_info = self.kb_manager.search(input_en, top_k=2)
        context_str = "\n".join([f"- {info}" for info in retrieved_info]) if retrieved_info else "No specific information found."
        
        # 5. Update History
        self.history.append({"role": "user", "content": input_en, "sentiment": sentiment})
        
        # 6. Sentiment-Aware & Context-Aware Prompting
        sentiment_instructions = {
            "positive": "The user is expressing positive emotions. Respond enthusiastically and affirmingly.",
            "negative": "The user is expressing negative emotions or frustration. Respond empathetically and offer concrete help.",
            "neutral": "The user's sentiment is neutral. Respond professionally, clearly, and concisely."
        }
        
        instruction = sentiment_instructions.get(sentiment, "Be a helpful, friendly, and empathetic AI assistant.")
        
        system_message = f"""You are a helpful and professional AI assistant.
Your goal is to answer the user's question based on the provided context from our knowledge base.

### Context from Knowledge Base:
{context_str}

### Guidelines:
- {instruction}
- Use the provided context to answer the question as accurately as possible.
- If the context doesn't contain the answer, politely state that you don't have that specific information in your current knowledge base, but offer to help with what you do know.
- Keep the response natural and conversational.
"""

        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add recent conversation history for context
        for turn in self.history[-4:]:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": turn['content']})
            elif "response" in turn:
                messages.append({"role": "assistant", "content": turn['response']})
        
        # Ensure the last message is the current user input
        if messages[-1]["role"] != "user" or messages[-1]["content"] != input_en:
            messages.append({"role": "user", "content": input_en})

        # 7. Generate Response using OpenAI
        assistant_response_en = "I'm sorry, I'm having trouble processing your request right now."
        try:
            response = self.openai_client.chat.completions.create(
                model=self.reasoning_model,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                assistant_response_en = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Reasoning Error: {e}")

        # 8. Translate back to user's language
        final_response = self.translate(assistant_response_en, 'en', lang_code)
        
        # Store in history
        self.history[-1]["response"] = assistant_response_en
        
        return {
            "detected_language": lang_code,
            "sentiment": sentiment,
            "retrieved_context": retrieved_info,
            "final_output": final_response
        }

if __name__ == "__main__":
    bot = MultilingualChatbot()
    
    # Clear and re-ingest fresh information for the demo
    if os.path.exists('/home/ubuntu/knowledge_base.index'):
        os.remove('/home/ubuntu/knowledge_base.index')
    if os.path.exists('/home/ubuntu/metadata.pkl'):
        os.remove('/home/ubuntu/metadata.pkl')
    bot.kb_manager = KnowledgeBaseManager()
    
    bot.kb_manager.add_information([
        "The Manus AI headquarters is located at 456 Innovation Way, San Francisco, CA 94105.",
        "Customer support is available 24/7 via our live chat or by emailing support@manus.ai.",
        "Our Pro plan costs $29.99 per month and includes unlimited multilingual support.",
        "The latest update (v2.0) introduced a dynamic knowledge base that updates every 24 hours.",
        "We currently support over 200 languages for both text and sentiment analysis."
    ])
    
    print("\n" + "="*50)
    print("   DYNAMIC KNOWLEDGE BASE MULTILINGUAL CHATBOT (v11)")
    print("="*50 + "\n")
    
    test_queries = [
        "What is the address of your headquarters?", # English
        "¿Cómo puedo hablar con alguien de soporte?", # Spanish (How can I talk to someone from support?)
        "Quel est le prix de l'abonnement Pro ?", # French (What is the price of the Pro subscription?)
        "Tell me about the latest update.", # English
        "Who is the president of the United States?" # Out of context
    ]
    
    for query in test_queries:
        print(f"You: {query}")
        res = bot.process_input(query)
        print(f"[Sentiment: {res['sentiment'].upper()}]")
        if res['retrieved_context']:
            print(f"[Context Matches: {len(res['retrieved_context'])}]")
        print(f"Chatbot: {res['final_output']}\n")
        print("-" * 50)
