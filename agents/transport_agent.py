from pydantic_ai import Agent
import mcp_client
from dataclasses import dataclass
from typing import List, Literal
from dotenv import load_dotenv
from model import get_openai_model


load_dotenv()
model = get_openai_model()

@dataclass
class TransportPreferences:
    max_travel_hours: int  # e.g., 2 means user is willing to travel up to 2 hours
    distance_preference: Literal["near", "moderate", "far", "very_far"]
    preferred_transport_modes: List[Literal["flight", "train", "bus", "cab", "car", "metro", "bike"]]
    ac_preference: Literal["ac", "non-ac", "any"]


system_prompt = f"""
You are a transport route and recommendation assistant that helps users plan travel to sports venues or events.

Use the following tools to search for transport options and shows the user the best routes according to his preferences:
{tool_descriptions}

Your responsibilities:
- Recommend suitable transport options and best routes based on user preferences such as:
    - maximum acceptable travel time (in hours),
    - comfort preference (AC or Non-AC),
    - preferred transport modes (bus, cab, train, flight, etc.),
    - distance willingness (e.g., near, far, very_far).
    
- If the distance is more than 400 km, you can **suggest flights**  if the user allows flight as a preferred transport mode.
- If distance is under 400 km, **avoid flights** and prefer road or train-based options unless user strongly prefers flight.
- Respect the user's AC/non-AC comfort choice and reject options that don't match (unless set to "any").
- Do not re-ask for already provided preferences.
- Do not confirm or book tickets — just suggest good options.

Give practical and concise suggestions, with clear reasoning based on user constraints.
"""

config_path = "./configs/transport.json"
async def get_transport_agent():
    client = mcp_client.MCPClient()
    client.load_servers(config_path)
    tools = await client.start()
    return client, Agent(model=get_openai_model(), system_prompt=system_prompt, deps_type=TransportPreferences, tools=tools)
