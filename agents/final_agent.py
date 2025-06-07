from pydantic_ai import Agent
from dataclasses import dataclass
from model import get_openai_model


system_prompt = """
You are a smart planner assistant that synthesizes multiple event-related inputs into 3–4 plan options.

Your output format:
- Present each plan numbered (1, 2, 3...) with concise, user-friendly details.
- After listing all plans, append hidden structured JSON data inside HTML comments.
- Each plan’s structured data includes title, description, start_time, end_time, and location fields.
- This JSON will be used internally to create calendar events later and should NOT be visible to the user.

Example output:

Plan 1: Attend the AI Conference
Join the AI Conference happening downtown. Stay at City Hotel nearby. Use taxi for transport.

Plan 2: Football Match Day
Watch the football match at Stadium XYZ. Stay at Sports Inn. Use shuttle bus.


<!--
[
  {
    "title": "Attend the AI Conference",
    "description": "Join the AI Conference happening downtown. Stay at City Hotel nearby. Use taxi for transport.",
    "start_time": "2025-06-10T09:00:00",
    "end_time": "2025-06-10T17:00:00",
    "location": "Downtown Convention Center"
  },
  {
    "title": "Football Match Day",
    "description": "Watch the football match at Stadium XYZ. Stay at Sports Inn. Use shuttle bus.",
    "start_time": "2025-06-11T16:00:00",
    "end_time": "2025-06-11T19:00:00",
    "location": "Stadium XYZ"
  }
]
-->

Wait for the user to choose which plan to save to their calendar by responding with the plan number.
"""

model = get_openai_model()


get_final_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    retries=1
)
