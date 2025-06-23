from PIL import Image
from smolagents import Tool, tool, OpenAIServerModel, DuckDuckGoSearchTool, CodeAgent, VisitWebpageTool
import requests
from io import BytesIO
import os
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # Joker image
]
images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" 
    }
    response = requests.get(url, headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)

model = OpenAIServerModel(model_id="gpt-4o")

agent = CodeAgent(
    tools = [
        DuckDuckGoSearchTool()
    ],
    max_steps=20,
    additional_authorized_imports=["selenium", "requests", "PIL", "io"],
    name="SmolImageAgent",
    description="An agent that can analyze images and search for information about them.",
    verbosity_level=2,
    model=model
)
response = agent.run(
    """
    Describe the costume and makeup that the comic character in these photos is wearing and return the description.
    Tell me if the guest is The Joker or Wonder Woman.


Thoughts: [your reasoning about how to solve the problem]
Code:
```py
# Your Python code here
```<end_code>

The code block MUST start with ```py on its own line and end with ```<end_code> on its own line.
    """,
    images=images
)