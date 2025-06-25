from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")

print(OPENAI_TOKEN)
llm = OpenAI(
    model="gpt-4o",
    api_key=OPENAI_TOKEN,
    temperature=0.7
)

response = llm.complete("hello, how are you?")
print(response)