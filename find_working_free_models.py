import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "https://github.com/coder/ai-agent",
        "X-Title": "AI Coding Agent CLI",
    }
)

free_models = [
    "meta-llama/llama-3-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "liquid/lfm-7b:free",
    "qwen/qwen-2-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",

]

for model in free_models:
    print(f"\n--- Testing model: {model} ---")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Hi"}
            ]
        )
        print(f"Success! Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Failed with exception: {type(e).__name__} - {str(e)}")
