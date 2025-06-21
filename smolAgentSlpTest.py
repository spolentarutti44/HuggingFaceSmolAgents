import os
import datetime
from smolagents import CodeAgent, tool,Tool,FinalAnswerTool, DuckDuckGoSearchTool, OpenAIServerModel, VisitWebpageTool
from config import OPENAI_API_KEY

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

@tool
def catering_tool(query: str) -> str:
    """
    A tool to search for catering services based on the query.
    Args:
        query (str): The search query for catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    best_service = max(services, key=services.get)

    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""
    
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }
    
    output_type = "string"

    def forward(self, category: str) -> str:
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        return themes.get(category.lower(), "No theme found for this category.")
    
# Create agent with OpenAI model and necessary tools
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        SuperheroPartyThemeTool(),
        catering_tool,
        FinalAnswerTool()
        ],
    max_steps=10,
    verbosity_level=2,
    model=OpenAIServerModel(model_id="gpt-3.5-turbo"),
    additional_authorized_imports=['datetime']
)
agent.run("Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")

# Run the agent
#agent.run("""
#    Alfred needs to prepare for the party. Here are the tasks:
#    1. Prepare the drinks - 30 minutes
#    2. Decorate the mansion - 60 minutes
#    3. Set up the menu - 45 minutes
#    4. Prepare the music and playlist - 45 minutes

#    If we start right now, at what time will the party be ready?
#   """)
