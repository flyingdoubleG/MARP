import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import HarmCategory

from vertexai.preview.generative_models import HarmBlockThreshold

import os
import google.generativeai as genai

def gemini_gen(message: str) -> str:
    config = genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(message)

    return response.text

def gemini_chat(message: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project="multi-agent-411823", location="us-central1")

    model = GenerativeModel("gemini-pro")
    chat = model.start_chat()

    response = chat.send_message(message)
    return response.text

def gemini_response(message: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project="multi-agent-411823", location="us-central1")

    # Load the model
    model = GenerativeModel("gemini-pro")

    config = {
        "temperature": 0.9,
        "top_p": 1
    }

    # Safety config
    safety_config = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    }
    
    # Query the model
    # response = model.generate_content(message, generation_config=config)
    response = model.generate_content(message, safety_settings=safety_config)
    return response.text


if __name__ == "__main__":
    # response = gemini_chat("Please tell me a joke.")
    # # response = gemini_response("Please tell me a joke.")
    # print(response)


    # Load your API key from the environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please export it in your .bash_profile")

    # Configure the API client
    genai.configure(api_key=api_key)

    # Define your prompt for Gemini Pro
    prompt = "Write a poem about the beauty of artificial intelligence."

    # Generate a response using a suitable Gemini Pro model
    response = genai.generate(
        model="ultra.1.0", 
        input=prompt,
        temperature=0.7  # Adjust for creativity 
    )

    # Print the generated poem
    print(response.generations[0].text) 