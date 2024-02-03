import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import HarmCategory

from vertexai.preview.generative_models import HarmBlockThreshold

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
    response = gemini_chat("Please tell me a joke.")
    # response = gemini_response("Please tell me a joke.")
    print(response)