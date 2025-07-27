from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from typing import Annotated, Dict, List, Any, Literal, Optional
from typing_extensions import TypedDict
from pydantic import ValidationError
from dataclasses import dataclass
import logfire
import asyncio
import os
import sys

from agents.information_agent import get_userinfo_agent, UserInfo
from agents.sports_venue_agent import get_venue_agent, VenuePreferences
from agents.transport_agent import get_transport_agent, TransportPreferences
from agents.stay_agent import get_stay_agent, StayPreferences
from agents.unified_event_agent import get_unified_event_agent, AllEventPreferences
from agents.final_agent import get_final_agent

from pydantic_ai.message import ModelMessage, ModelMessagesTypeAdapter


class State(TypedDict):
    # User details/info and chat messages
    user_input: str
    messages: Annotated[List[bytes], lambda x, y: x+y]
    user_details: Dict[str, Any]
    
    # Venue Preferences    
    preferred_games: List[str]
    preferred_timeslot: str
    gym_availability: bool
    
    # Transport Preferences
    max_travel_hours: int 
    distance_preference: Literal["near", "moderate", "far", "very_far"]
    preferred_transport_modes: List[Literal["flight", "train", "bus", "car", "metro"]]
    ac_preference: Literal["ac", "non-ac", "any"]
    
    # Stay Preferences
    room_type: Literal["hotel", "pg/hostel","any"]
    ac_preference: Literal["ac", "non-ac", "any"]
    preferred_amenities: List[str]
    max_budget: int
    
    # All Events Preferences

    # GameEventPreferences:
    format: Literal["online", "offline", "hybrid"]
    event_type: Literal["team", "solo", "any"]
    location_scope: Literal["same_city", "inter_city", "any"]
    competitive_level: Literal["casual", "competitive", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

    # FitnessEventPreferences:
    fitness_type: Literal["yoga", "gym", "zumba", "pilates", "any"]
    format: Literal["offline", "online", "hybrid"]
    location_scope: Literal["nearby", "citywide", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

    #TechEventPreferences:
    topic: str  # e.g., "machine learning"
    format: Literal["online", "offline", "hybrid"]
    location_scope: Literal["same_city", "inter_city", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

    #GeneralEventPreferences:
    interest_area: str  # e.g., "music", "networking"
    format: Literal["online", "offline", "hybrid"]
    location_scope: Literal["same_city", "inter_city", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None
    
    
async def collect_user_info(state: State, writer) -> Dict[str, Any]:
    # Get the user information
    user_input = state["user_input"]

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))    
    
    # Call the info gathering agent
    # result = await info_gathering_agent.run(user_input)
    async with get_userinfo_agent.run_stream(user_input, message_history=message_history) as result:
        curr_response = ""
        async for message, last in result.stream_structured(debounce_by=0.01):  
            try:
                if last and not user_details.response:
                    raise Exception("Incorrect travel details returned by the agent.")
                user_details = await result.validate_structured_result(  
                    message,
                    allow_partial=not last
                )
            except ValidationError as e:
                continue

            if user_details.response:
                writer(user_details.response[len(curr_response):])
                curr_response = user_details.response  

    # Return the response asking for more details if necessary
    data = await result.get_data()
    return {
        "user_details": data.model_dump(),
        "messages": [result.new_messages_json()]
    }    
    
def get_chat_message(state: State):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "user_input": value
    }  

def route_userinfo(state: State):
    
    user_details = state["user_details"]
    intent = user_details["intent"]
    
    if not user_details.get("all_details_given"):
        return "get_chat_message"
    
    else:
        if intent.endswith("_venue"):  # Check if the intent ends with "_venue"
            return "get_venue_agent"
        elif intent.endswith("_event"):  # Check if the intent ends with "_event"
            return "get_unified_event_agent"
        
async def get_venue_agent(state: State, writer) -> Dict[str, Any]:
    writer("\n Searching for venues...\n")
    user_details = state["user_details"]
    intent = user_details["intent"]
    location = user_details["location"]
    start_date = user_details["user_date_first"]
    end_date = user_details["user_date_last"]:Optional[str]
    game_name = user_details["game_name"]
    
    prompt = f"I need venue recommendations for {game_name} for the location '{location}' from {start_date} to {end_date}"
    
    # Call the venue agent
    output = await get_venue_agent.run(prompt, deps=venue_dependencies)
    
    return {"venue_output": output.data}

async def get_unified_event_agent(state: State, writer) -> Dict[str, Any]:
    writer("\n Searching for events based on your interests...\n")
    user_details = state["user_details"]
    intent = user_details["intent"]
    location = user_details["location"]
    start_date = user_details["user_date_first"]
    end_date = user_details["user_date_last"]:Optional[str]
    location_scope = event_dependencies["location_scope"]
    
    # intent-specific query building
    if intent == "book_game_event":
        game_name = user_details["game_name"]
        category = game_name
    elif intent == "book_fitness_event":
        fitness_type = user_details["fitness_type"]
        category = fitness_type
    else:
        event_name = user_details["event_name"]
        category = event_name
        
    prompt = (
        " Suggest 2-3 interesting events related to {category} between these dates: {start_date} and {end_date}."
        " Location of user is {location}. Use the location_scope:({location_scope}) to find the events"
        " Check in {event_dependencies} whether the user prefers online or offline events."
        " Each event should include title/name of event, location of event, format of event (online/offline), event_start_date, event_end_date(Optional, if one-day event), and a description of the event."
        " Return the response as JSON list of events."
    )
    
    # Call/ Run the unified event agent
    output = await get_unified_event_agent.run(prompt, deps=event_dependencies)
    
    # Example output.data expected:
    # [
    #   {
    #       "title": "AI Conference",
    #       "location": "New York",
    #       "format": "online",
    #       "event_start_date": "2023-06-01",
    #       "event_end_date": "2023-06-03",
    #       "description": "AI conference on "
    #    },
    #    ...
    # ]
    return {"event_output": output.data}

        
def route_to_all(state: State):
    return ["get_stay_agent", "get_transport_agent"]
 
async def get_stay_agent(state: State):
    
    writer("\n Evaluating if stay suggestions are needed...\n")
    
    user_details = state["user_details"]
    is_online = user_details.get("event_mode", "offline") == "online"
    start_date = user_details["user_date_first"]
    end_date = user_details.get("user_date_last")
    location = user_details["location"]

    # Skip stay recommendation if event is online or single-day
    if is_online or not end_date or start_date == end_date:
        writer("Event is online or single-day. Skipping stay suggestions.\n")
        return {"stay_output": "No stay needed"}

    writer("\n Searching for stay options...\n")

    # Get the agent and tools
    #client, agent = await get_stay_agent()

    # prompt the llm
    prompt = f"Suggest good stay options in/near {location} for event dates {start_date} to {end_date}"

    # Run agent
    output = await get_stay_agent.run(prompt, deps=StayPreferences)

    return {"stay_output": output.data}
    
    

async def get_transport_agent(state: State):
    writer("\n Evaluating if transport suggestions are needed...\n")
    
    user_details = state["user_details"]
    is_online = user_details.get("event_mode", "offline") == "online"
    location = user_details["location"]
    origin = user_details.get("origin")  # Optional: Where the user is coming from
    start_date = user_details["user_date_first"]

    # Skip transport if event is online or no origin is given
    if is_online or not origin or origin.lower() == location.lower():
        writer("Event is online or user is local. Skipping transport suggestions.\n")
        return {"transport_output": "No transport needed"}

    writer("\n Searching for transport options...\n")

    # Get the agent and tools
    client, agent = await get_transport_agent()

    # Prompt for LLM
    prompt = f"Suggest transport options from {origin} to {location} for arrival by {start_date}"

    # Run agent with appropriate schema
    output = await agent.run(prompt, deps=TransportPreferences)

    return {"transport_output": output.data}
    
    
async get_final_agent(state: State, writer) -> Dict[str, Any]:
    user_details = state['user_details']
    event_results = state['event_output']
    venue_results = state['venue_output']
    stay_results = state['stay_output']
    transport_results = state['transport_output']
    
    prompt = f"""
    I am planning to go from my location ({user_details.get('location)}) to a
    
    # Call the final agent
    output = await get_final_agent.run(prompt, deps=FinalPreferences)
    
    return {"final_output": output.data}
    
    

def sports_events_agent_graph():
    "Building and returning the graph"
    graph = StateGraph(State)
    
    graph.add_node("collect_user_info", collect_user_info())
    graph.add_node("get_chat_message", get_chat_message())
    graph.add_node("get_venue_agent", get_venue_agent())
    graph.add_node("get_transport_agent", get_transport_agent())
    graph.add_node("get_stay_agent", get_stay_agent())
    graph.add_node("get_unified_event_agent", get_unified_event_agent())
    graph.add_node("get_final_agent", get_final_agent())
    
    
    # edges
    graph.add_edge(START, "collect_user_info")
    
    graph.add_conditional_edges("collect_user_info", route_userinfo, ["get_chat_message", "get_venue_agent", "get_unified_event_agent"])
    
    graph.add_edge("get_chat_message", "get_userinfo_agent")
    graph.add_conditional_edges("get_venue_agent",route_to_all, ["get_transport_agent", "get_stay_agent"])
    graph.add_conditional_edges("get_unified_event_agent",route_to_all, ["get_transport_agent", "get_stay_agent"])
    graph.add_edge("get_transport_agent", "get_final_agent")
    graph.add_edge("get_stay_agent", "get_final_agent")
   
    graph.add_edge(get_final_agent(), END)
    
    memory = MemorySaver()
    return graph.compile(checkpoint=memory)

sports_event_agent_graph = sports_events_agent_graph()
    

    
    
    
    
