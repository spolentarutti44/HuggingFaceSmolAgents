from smolagents import Tool, tool, OpenAIServerModel,InferenceClientModel, DuckDuckGoSearchTool, CodeAgent, VisitWebpageTool
from typing import Optional, Tuple
from config import OPENAI_API_KEY
import math
import os
import pandas as pd
from smolagents.utils import encode_image_base64, make_image_url
from PIL import Image
import matplotlib

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
)->float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """
    def to_radians(degrees):
        return degrees * (math.pi / 180.0)
    
    lat, long = map(to_radians, origin_coords)
    lat2, long2 = map(to_radians, destination_coords)
    EARTH_RADIUS = 6371.0  # km

    dlong = long2 - long
    dlat = lat2 - lat
    a = math.sin(dlat / 2)**2 + math.cos(lat) * math.cos(lat2) * math.sin(dlong / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    DISTANCE = EARTH_RADIUS * c

    actual_distance = DISTANCE * 1.1  # Adjust for cargo flight path inefficiencies
    time = (actual_distance / cruising_speed_kmh)+1.0      # Add 1 hour for takeoff and landing procedures

    return round(time,2)

def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4", 
    max_tokens=1500)
    file_path = "image.png"
    assert os.path.exists(file_path), "Image file does not exist"
    image = Image.open(file_path)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Reasoning and plot check output:", output)
    if "fail" in output.lower():
        print("Reasoning check failed. Please review the agent's reasoning.")


# Increase max_tokens to allow for complete responses
model = OpenAIServerModel(
    model_id="gpt-3.5-turbo", 
    temperature=0.0, 
    max_tokens=1500
)

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
    model=model,
    additional_authorized_imports=["pandas"],
    max_steps=10,
    name="WebAgent",
    verbosity_level=0,
    description="An agent that can search the web, visit webpages, and calculate cargo travel times between locations."
)

manager_agent = CodeAgent(
        model=OpenAIServerModel("gpt-3.5-turbo"),
        tools=[calculate_cargo_travel_time],
        managed_agents=[web_agent],
        name="ManagerAgent",
        description="A manager agent that can delegate tasks to other agents and manage their execution.",
        additional_authorized_imports=[
            "pandas",
            "geopandas",
            "shapely",
            "plotly",
            "json",
            "numpy",
            "matplotlib",
        ],
        planning_interval=5,
        verbosity_level=2,
        max_steps=15,
        final_answer_checks=[check_reasoning_and_plot]
        )
#manager_agent.visualize() 

manager_agent.run("""
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
                  

Thoughts: [your reasoning about how to solve the problem]
Code:
```py
# Your Python code here
```<end_code>

The code block MUST start with ```py on its own line and end with ```<end_code> on its own line.
""")
manager_agent.python_executor.state["fig"]

