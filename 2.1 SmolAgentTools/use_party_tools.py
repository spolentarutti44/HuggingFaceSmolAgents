import os
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, OpenAIServerModel
from party_tools import suggest_menu, catering_tool,image_generation_tool, SuperheroPartyThemeTool
from config import OPENAI_API_KEY

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Create agent with OpenAI model and necessary tools
agent = ToolCallingAgent(
    tools=[
        DuckDuckGoSearchTool(),
        suggest_menu,
        image_generation_tool,
        SuperheroPartyThemeTool(),
        catering_tool
    ],
    model=OpenAIServerModel(model_id="gpt-3.5-turbo")
)

# Run the agent
result = agent.run("Give me a menu suggestion for a formal party and recommend a catering service.")
print(result)

# Try another query with the superhero theme tool
result = agent.run("What would be a good theme for a villain masquerade superhero party?")
print(result)