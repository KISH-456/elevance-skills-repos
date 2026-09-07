from multilingual_chatbot_v2 import MultilingualChatbot
import time

def run_tests():
    print("Starting Multilingual Chatbot Verification Tests...")
    bot = MultilingualChatbot()
    
    test_cases = [
        {
            "name": "Language Identification (Spanish)",
            "input": "¿Cómo estás hoy?",
            "expected_lang": "es"
        },
        {
            "name": "Mixed-Language Input",
            "input": "Hola, can you tell me what time it is?",
            "expected_lang": "es" # langdetect might pick es or en
        },
        {
            "name": "Context Preservation across Language Switch",
            "input": "My favorite color is blue.",
            "expected_lang": "en"
        },
        {
            "name": "Context Check (French)",
            "input": "Quelle est ma couleur préférée ?", # "What is my favorite color?"
            "expected_lang": "fr"
        },
        {
            "name": "Cross-lingual Reasoning (German)",
            "input": "Was ist die Hauptstadt von Frankreich?", # "What is the capital of France?"
            "expected_lang": "de"
        }
    ]
    
    for case in test_cases:
        print(f"\nRunning Test: {case['name']}")
        print(f"Input: {case['input']}")
        start_time = time.time()
        result = bot.process_input(case['input'])
        end_time = time.time()
        
        print(f"Detected Lang: {result['detected_language']}")
        print(f"Internal EN:    {result['internal_response']}")
        print(f"Final Output:   {result['final_output']}")
        print(f"Processing Time: {end_time - start_time:.2f}s")
        
    print("\nTests Completed.")

if __name__ == "__main__":
    run_tests()
