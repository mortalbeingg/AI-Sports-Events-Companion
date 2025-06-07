from pydantic_ai import Agent
import mcp_client
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from model import get_openai_model

load_dotenv()

model = get_openai_model()

@dataclass
class VenuePreferences:
    preferred_games: List[str]
    preferred_timeslot: str
    gym_availability: bool
    

system_prompt = f"""
You are a sports venue search assistant that helps users discover venues for their preferred sports.

Use the following tools to perform searches:
{tool_descriptions}

Your responsibilities:
- Recommend suitable sports venues based on preferences like game type, location, time slot.
- Use context dependencies like preferred games or time slots.
- Do not re-ask for already provided information.
- Do not confirm bookings — only search and recommend relevant venues.

Respond clearly and helpfully, with a strong reasoning for your suggestions.
"""

    
async def get_venue_agent():
    client = mcp_client.MCPClient()
    client.load_servers(str(CONFIG_FILE))
    tools = await client.start()
    return client, Agent(model=get_openai_model(), system_prompt=system_prompt, deps_type=UserPreferences, tools=tools)   