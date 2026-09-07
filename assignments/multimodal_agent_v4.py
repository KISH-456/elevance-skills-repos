import os
import json
import base64
from openai import OpenAI
from typing import List, Dict, Any, Optional

class MultimodalAgent:
    """
    A multi-modal AI assistant that uses an agentic reasoning loop (Plan-Execute-Verify)
    to process text and image inputs, maintain context, and validate responses.
    """
    def __init__(self, model="gpt-4.1-mini"):
        print(f"Initializing Multimodal Agent with {model}...")
        self.client = OpenAI()
        self.model = model
        self.history = [] # Stores {'role': 'user'/'assistant', 'content': 'text'}
        self.persistent_images = [] # Stores base64 encoded images that persist across turns

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def add_image(self, image_path: str):
        """Adds an image to the persistent context."""
        if os.path.exists(image_path):
            base64_image = self._encode_image(image_path)
            self.persistent_images.append(base64_image)
            print(f"Image added to persistent context: {image_path}")
        else:
            print(f"Error: Image path {image_path} does not exist.")

    def clear_images(self):
        """Clears all persistent images from the context."""
        self.persistent_images = []
        print("All images cleared from persistent context.")

    def _call_llm(self, messages: List[Dict[str, Any]], response_format: Optional[str] = None) -> str:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1000
            }
            if response_format == "json":
                params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return ""

    def process_query(self, user_query: str) -> str:
        """
        The main agentic reasoning loop: Plan -> Execute (Extract/Reason) -> Verify -> Respond.
        """
        print(f"\nProcessing User Query: {user_query}")
        
        # 1. PLANNING PHASE
        plan = self._plan(user_query)
        if not plan: plan = "Proceed with direct reasoning."
        print(f"[Plan]: {plan}")

        # 2. EXECUTION PHASE (Visual Extraction & Reasoning)
        evidence = self._extract_evidence(user_query, plan)
        if not evidence: evidence = "No evidence could be extracted."
        print(f"[Evidence]: {evidence}")

        # 3. VERIFICATION PHASE
        is_valid, verification_notes = self._verify(user_query, evidence)
        print(f"[Verification]: Valid={is_valid}, Notes={verification_notes}")

        if not is_valid:
            print("Evidence insufficient or ambiguous. Attempting refinement...")
            evidence = f"{evidence}\nAdditional Verification Gaps: {verification_notes}"

        # 4. FINAL RESPONSE GENERATION
        final_response = self._generate_final_response(user_query, evidence)
        if not final_response:
            final_response = "I apologize, but I\\'m having trouble processing that right now."
        
        # Update History
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": final_response})
        
        return final_response

    def _plan(self, query: str) -> str:
        system_msg = (
            "You are a strategic planner for a multi-modal AI. Given a user query and any existing images in the persistent context, "
            "outline the steps needed to answer accurately. "
            "Determine if visual extraction from the persistent images is needed, or if the query relies solely on conversational history. "
            "If visual extraction is needed, specify what details to extract. If not, state that no visual extraction is required."
        )
        messages = [{"role": "system", "content": system_msg}]
        # Add history for context
        for turn in self.history[-4:]:
            messages.append(turn)
        messages.append({"role": "user", "content": f"Query: {query}\nNumber of persistent images: {len(self.persistent_images)}"})
        return self._call_llm(messages)

    def _extract_evidence(self, query: str, plan: str) -> str:
        """
        Executes targeted visual extraction based on the plan, or summarizes history if plan indicates.
        """
        if "no visual extraction is needed" in plan.lower() or not self.persistent_images:
            # If no visual extraction is needed or no images are present, summarize history as evidence
            history_summary = "\n".join([f"{t["role"]}: {t["content"]}" for t in self.history[-5:]])
            return f"Conversational History Summary:\n{history_summary}\n\nBased on the plan, no visual extraction was performed as no images were provided or deemed necessary for this query."

        system_msg = (
            "You are an expert visual analyst. Your task is to perform targeted extraction "
            "of information from images based on a provided plan. "
            "Rules:\n"
            "1. Be extremely precise and objective.\n"
            "2. Cite specific visual evidence (e.g., \'In the top-right corner...\', \'The text on the label says...\').\n"
            "3. If multiple images are provided, clearly distinguish which image you are referring to.\n"
            "4. If something in the plan is not visible, explicitly state that."
        )
        
        content = [{"type": "text", "text": f"Plan: {plan}\nUser Query: {query}\n\nPerform the extraction based on the above."}]
        for i, img in enumerate(self.persistent_images):
            content.append({"type": "text", "text": f"--- Persistent Image {i+1} ---"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content}
        ]
        return self._call_llm(messages)

    def _verify(self, query: str, evidence: str) -> (bool, str):
        """
        Validates the extracted evidence against the user query.
        """
        system_msg = (
            "You are a meticulous verification specialist. Evaluate if the evidence is "
            "sufficient to answer the user query accurately.\n"
            "Respond in JSON: {\"is_valid\": boolean, \"notes\": \"string\"}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"User Query: {query}\nExtracted Evidence: {evidence}\n\nVerify the evidence."}
        ]
        verification_json = self._call_llm(messages, response_format="json")
        try:
            data = json.loads(verification_json)
            return data.get("is_valid", False), data.get("notes", "")
        except:
            return False, "Verification failed to parse or invalid JSON."

    def _generate_final_response(self, query: str, evidence: str) -> str:
        system_msg = "You are a helpful, evidence-based multi-modal assistant. Generate a final response to the user based on the query and the verified evidence. Maintain conversational context."
        messages = [{"role": "system", "content": system_msg}]
        for turn in self.history[-4:]:
            messages.append(turn)
        messages.append({"role": "user", "content": f"Query: {query}\nEvidence: {evidence}"})
        return self._call_llm(messages)

if __name__ == "__main__":
    agent = MultimodalAgent()
    print("Multimodal Agent V4 Ready.")

    # Example usage:
    # To test with an image, first create a dummy image file
    from PIL import Image
    dummy_image_path = "dummy_image.png"
    img = Image.new("RGB", (60, 30), color = "red")
    img.save(dummy_image_path)

    # Text-only queries
    response1 = agent.process_query("Hello, I am testing your memory.")
    print(f"Agent Response 1: {response1}")

    response2 = agent.process_query("What was the first thing I said?")
    print(f"Agent Response 2: {response2}")

    response3 = agent.process_query("Can you summarize our conversation so far?")
    print(f"Agent Response 3: {response3}")

    # Image processing
    agent.add_image(dummy_image_path)
    response4 = agent.process_query("What color is the object in the image?")
    print(f"Agent Response 4: {response4}")

    # Image processing with context (should remember the image)
    response5 = agent.process_query("Can you describe the shape of the object?")
    print(f"Agent Response 5: {response5}")

    # Clear images and ask another question
    agent.clear_images()
    response6 = agent.process_query("Are there any images in our current context?")
    print(f"Agent Response 6: {response6}")

    # Clean up dummy image
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
