from pydantic_ai import Agent
import mcp_client
from dataclasses import dataclass
from typing import Literal, Optional, Union
from dotenv import load_dotenv
from model import get_openai_model

load_dotenv()
model = get_openai_model()

# INTENT-SPECIFIC PREFERENCES

@dataclass
class GameEventPreferences:
    format: Literal["online", "offline", "hybrid"]
    event_type: Literal["team", "solo", "any"]
    location_scope: Literal["same_city", "inter_city", "any"]
    competitive_level: Literal["casual", "competitive", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

@dataclass
class FitnessEventPreferences:
    fitness_type: Literal["yoga", "gym", "zumba", "pilates", "any"]
    format: Literal["offline", "online", "hybrid"]
    location_scope: Literal["nearby", "citywide", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

@dataclass
class TechEventPreferences:
    topic: str  # e.g., "machine learning"
    format: Literal["online", "offline", "hybrid"]
    location_scope: Literal["same_city", "inter_city", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

@dataclass
class GeneralEventPreferences:
    interest_area: str  # e.g., "music", "networking"
    format: Literal["online", "offline", "hybrid"]
    location_scope: Literal["same_city", "inter_city", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

# UNION for all types of preferences
AllEventPreferences = Union[
    GameEventPreferences,
    FitnessEventPreferences,
    TechEventPreferences,
    GeneralEventPreferences,
]

# SYSTEM PROMPT
system_prompt = f"""
You are a smart event discovery assistant that handles multiple types of events.

Supported intents:
- book_game_event → for sports matches, tournaments, friendly games
- book_fitness_event → for fitness classes or sessions
- book_tech_event → for conferences, workshops, tech meetups
- book_general_event → for other social, cultural, or personal interest events

Use the right preferences based on intent:

1. For 'book_game_event':
    - format: online/offline/hybrid
    - event_type: team/solo
    - location_scope
    - competitive_level
    - is_paid, budget_if_paid

2. For 'book_fitness_event':
    - fitness_type (e.g., yoga)
    - format (e.g., in_person)
    - location_scope
    - is_paid, budget_if_paid

3. For 'book_tech_event':
    - topic (e.g., AI, blockchain)
    - format
    - location_scope
    - is_paid, budget_if_paid

4. For 'book_general_event':
    - interest_area (e.g., music, networking)
    - format
    - location_scope
    - is_paid, budget_if_paid


Use the following tools to search for events options:
{tool_descriptions}

Guidelines:
- Always apply budget filters if the user prefers 'paid' events and gives a budget.
- Never suggest paid events if 'free' is specified.
- Suggest best-fit events with reasons based on the given preferences.
- Don’t reconfirm known preferences. Don’t book — only discover.

You’ll receive the correct preference schema dynamically.
"""


INTENT_CONFIG_MAP = {
    "book_game_event": "configs/event_game.json",
    "book_fitness_event": "configs/event_fitness.json",
    "book_tech_event": "configs/event_tech.json",
    "book_general_event": "configs/event_general.json",
}

INTENT_PREFS_CONFIG_MAP = {
    "book_game_event": AllEventPreferences[GameEventPreferences],
    "book_fitness_event": AllEventPreferences[FitnessEventPreferences],
    "book_tech_event": AllEventPreferences[TechEventPreferences],
    "book_general_event": AllEventPreferences[GeneralEventPreferences],
}

# UNIFIED EVENT AGENT 

async def get_unified_event_agent(intent:str):
    config_path = INTENT_CONFIG_MAP.get(intent)
    if not config_path or not Path(config_path).exists():
        raise ValueError(f"No config found for the required intent: {intent}")
    
    client = mcp_client.MCPClient()
    client.load_servers(config_path)
    tools = await client.start()
    return client, Agent(
        model=get_openai_model(),
        system_prompt=system_prompt,
        deps_type=INTENT_PREFS_CONFIG_MAP.get(intent),
        tools=tools,
        system_prompt=system_prompt
    )
