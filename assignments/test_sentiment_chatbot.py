from multilingual_chatbot_v8 import MultilingualChatbot
import time

def run_sentiment_tests():
    print("Starting Sentiment-Aware Multilingual Chatbot Verification Tests...")
    bot = MultilingualChatbot()
    
    test_cases = [
        {
            "name": "Positive Sentiment (English)",
            "input": "I love this chatbot, it's so helpful!",
            "expected_sentiment": "positive"
        },
        {
            "name": "Negative Sentiment (Spanish)",
            "input": "Estoy muy enfadado con el servicio, es horrible.",
            "expected_sentiment": "negative"
        },
        {
            "name": "Neutral Sentiment (French)",
            "input": "Quelle heure est-il à Paris ?",
            "expected_sentiment": "neutral"
        },
        {
            "name": "Context & Sentiment Switch",
            "input": "Actually, I'm feeling much better now, thank you for the help.",
            "expected_sentiment": "positive"
        }
    ]
    
    for case in test_cases:
        print(f"\nRunning Test: {case['name']}")
        print(f"Input: {case['input']}")
        start_time = time.time()
        result = bot.process_input(case['input'])
        end_time = time.time()
        
        print(f"Detected Lang: {result['detected_language']}")
        print(f"Sentiment:     {result['sentiment'].upper()}")
        print(f"Internal EN:    {result['internal_response']}")
        print(f"Final Output:   {result['final_output']}")
        print(f"Processing Time: {end_time - start_time:.2f}s")
        
    print("\nTests Completed.")

if __name__ == "__main__":
    run_sentiment_tests()
