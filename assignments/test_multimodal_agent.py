from multimodal_agent_v4 import MultimodalAgent
import os
from PIL import Image

def run_tests():
    print("Starting Multimodal Agent Verification Tests...")
    agent = MultimodalAgent()
    
    # Create a dummy image for testing
    dummy_image_path = "dummy_image.png"
    img = Image.new("RGB", (60, 30), color = "red")
    img.save(dummy_image_path)
    
    # Test 1: Text-only context check
    print("\n--- Test 1: Text-only Context ---")
    res1 = agent.process_query("Hello, I\'m testing your reasoning capabilities.")
    print(f"Agent Response 1: {res1}")
    
    # Test 2: Ambiguity Handling (No images provided yet, but query implies one)
    print("\n--- Test 2: Ambiguity Handling ---")
    res2 = agent.process_query("What is shown in the image I just sent?")
    print(f"Agent Response 2: {res2}")
    
    # Test 3: Context Retention (text-only)
    print("\n--- Test 3: Context Retention ---")
    res3 = agent.process_query("What was the first thing I said to you?")
    print(f"Agent Response 3: {res3}")

    # Test 4: Image processing (add image to persistent context)
    print("\n--- Test 4: Image Processing ---")
    agent.add_image(dummy_image_path)
    res4 = agent.process_query("What color is the object in the image?")
    print(f"Agent Response 4: {res4}")

    # Test 5: Image processing with persistent context (should remember the image)
    print("\n--- Test 5: Image Processing with Persistent Context ---")
    res5 = agent.process_query("Can you describe the shape of the object?")
    print(f"Agent Response 5: {res5}")

    # Test 6: Clear images and ask another question
    print("\n--- Test 6: Clear Images and Re-query ---")
    agent.clear_images()
    res6 = agent.process_query("Are there any images in our current context?")
    print(f"Agent Response 6: {res6}")

    # Clean up dummy image
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)

    print("\nTests Completed.")

if __name__ == "__main__":
    run_tests()
