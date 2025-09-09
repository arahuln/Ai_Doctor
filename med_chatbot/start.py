from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os

load_dotenv()

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
nvidia_model_name = os.getenv("NVIDIA_MODEL_NAME")

text_input = input("Enter the input: ")

llm = ChatNVIDIA(
    model_name = nvidia_model_name,
    api_key = nvidia_api_key,
)

while True:
    if text_input.lower() == "exit":
        break
    else:
        result = llm.invoke(text_input)
        if result is not None:
            print(result.response)


