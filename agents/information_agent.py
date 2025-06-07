from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional
from datetime import date
from model import get_openai_model
import json 
import os
import sys


model = get_openai_model()


class UserInfo(BaseModel):
    intent: str = Field(description="The user's booking intent. Must be one of: 'book_game', 'book_event', or 'book_fitness'.")
    # for 'book_game'
    game_name: str = Field(description="Name of the sport or game the user wants to book a venue for (e.g., football, badminton).")

    # For 'book_event'
    event_name: str = Field(description="Topic or keyword for the event the user wants to attend (e.g., data science, AI conference).")

    # For 'book_fitness'
    fitness_type: str = Field(description="Type of fitness class or session (e.g., yoga, pilates, gym workout).")

    location: str = Field(description="City or location where the booking is intended.")
    user_date_first: str = Field(description="Start date of the booking or event (in YYYY-MM-DD format).")
    user_date_last: Optional[str] = Field(description="End date if the user provided a date range (in YYYY-MM-DD format).")
    all_details_given: bool = Field(description="True if all required fields are filled for the detected intent.")


today = date.today().isoformat()

system_prompt = f"""
You are an assistant that classifies booking intent and extracts details.

Today's date is {today}. Interpret relative phrases like "this Friday" or "next weekend" accordingly.

Your task:
- Classify intent as one of the following:
  - book_game_venue → when the user wants to book a ground, court, or sports facility to play
  - book_game_event → when the user wants to attend a sports event or tournament
  - book_fitness_event → when the user wants to attend a fitness-related event like yoga or gym workshop
  - book_tech_event → when the user wants to attend a tech-related event (e.g., data science, AI conference)
  - book_general_event → when the user wants to attend a general or non-tech event (e.g., music meetup, business event)

Collect the following fields depending on intent:
- book_game_venue:
    - game_name (e.g., badminton, football)
    - location
    - user_date_first (the date they want to book)

- book_game_event:
    - game_name (e.g., cricket, kabaddi)
    - location
    - user_date_first (date of the match or sports event)

- book_fitness_event:
    - fitness_type (e.g., yoga, zumba)
    - location
    - user_date_first (date of the session)

- book_tech_event:
    - event_name (e.g., data science conference, AI summit)
    - location
    - user_date_first (date of the event)

- book_general_event:
    - event_name (e.g., business networking, music festival)
    - location
    - user_date_first (date of the event)

If the user provides a date range (e.g., "between June 10 and June 12"):
- Set user_date_first to start date
- Set user_date_last to end date

If only one date is mentioned, set user_date_first and leave user_date_last as null.

Only set 'all_details_given' to true if:
- intent is clearly identified AND
- all required fields for that intent are present.
"""


get_userinfo_agent = Agent(
    model,
    output_type=user_info_output,
    system_prompt=system_prompt,
    retries=2
)