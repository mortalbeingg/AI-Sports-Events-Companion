from pydantic_ai import Agent
import mcp_client
from dataclasses import dataclass
from typing import List, Literal
from dotenv import load_dotenv
from model import get_openai_model

load_dotenv()
model = get_openai_model()

# Dependencies: user stay preferences (can be extended later)
@dataclass
class StayPreferences:
    room_type: Literal["hotel", "pg/hostel","any"]
    ac_preference: Literal["ac", "non-ac", "any"]
    preferred_amenities: List[str]
    max_budget: int

# Load MCP tools from hotel/PG related config
config_path = "./configs/stay.json"



# Function to dynamically build prompt with tool descriptions
def build_tool_descriptions(tools: list) -> str:
    return "\n".join(
        f"- {tool.name}: {tool.description or 'No description provided'}"
        for tool in tools
    )

async def get_stay_agent():
    client = mcp_client.MCPClient()
    client.load_servers(config_path)
    tools = await client.start()
    tool_descriptions = build_tool_descriptions(tools),
    system_prompt = f"""
You are a stay search assistant that helps users find accommodation options near sports venues or events **only when the event spans multiple days**.

Use the following tools to search for stay options:
{tool_descriptions}

Responsibilities:
- Suggest accommodation **only if the event is more than one day long**.
- Use context preferences such as:
    - accommodation type (hotel, PG, hostel),
    - AC/non-AC preference,
    - budget level (budget, mid-range, luxury),
    - specific amenities like Wi-Fi, parking, etc.

Instructions:
- Do **not** suggest any stays if the event is a single-day event or an online event.
- Never confirm bookings, only recommend based on preferences.
- Respect all user preferences from context and suggest top 2–3 options only.
- If no options exactly match, explain tradeoffs (e.g., "This PG has most amenities but not AC").

Be clear, practical, and user-friendly in your suggestions.
"""
    
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        deps_type=StayPreferences,
        tools=tools,
        retries=2
    )
    return client, agent
