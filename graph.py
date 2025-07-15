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
    event_format: Literal["online", "offline", "hybrid"]
    event_type: Literal["team", "solo", "any"]
    location_scope: Literal["same_city", "inter_city", "any"]
    competitive_level: Literal["casual", "competitive", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

    # FitnessEventPreferences:
    fitness_type: Literal["yoga", "gym", "zumba", "pilates", "any"]
    session_format: Literal["in_person", "online", "hybrid"]
    location_scope: Literal["nearby", "citywide", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

    #TechEventPreferences:
    topic: str  # e.g., "machine learning"
    format: Literal["online", "offline", "hybrid"]
    scope: Literal["same_city", "inter_city", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None

    #GeneralEventPreferences:
    interest_area: str  # e.g., "music", "networking"
    format: Literal["online", "offline", "hybrid"]
    location_scope: Literal["same_city", "inter_city", "any"]
    is_paid: Literal["free", "paid", "any"]
    budget_if_paid: Optional[float] = None
    
def sports_events_agent_graph():
    "Building and returning the graph"
    graph = StateGraph(State)
    
    graph.add_node("get_userinfo_agent", get_userinfo_agent())
    graph.add_node("get_chat_message", get_chat_message())
    graph.add_node("get_venue_agent", get_venue_agent())
    graph.add_node("get_transport_agent", get_transport_agent())
    graph.add_node("get_stay_agent", get_stay_agent())
    graph.add_node("get_unified_event_agent", get_unified_event_agent())
    graph.add_node("get_final_agent", get_final_agent())
    
    # edges
    graph.add_edge(START, "get_userinfo_agent")
    
    graph.add_conditional_edges("get_userinfo_agent", route_to_one)
    
    graph.add_edge("get_chat_message", "get_userinfo_agent")
    graph.add_conditional_edges("get_venue_agent",route_to_all, ["get_transport_agent", "get_stay_agent"])
    graph.add_conditional_edges("get_unified_event_agent",route_to_all, ["get_transport_agent", "get_stay_agent"])
    graph.add_edge("get_transport_agent", "get_final_agent")
    graph.add_edge("get_stay_agent", "get_final_agent")
   
    graph.add_edge(get_final_agent(), END)
    
    memory = MemorySaver()
    return graph.compile(checkpoint=memory)

sports_event_agent_graph = sports_events_agent_graph()
    

    
    
    
    
