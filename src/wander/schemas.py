"""Data shapes shared by the whole app. Spark's structured output is validated against these."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TripBrief(BaseModel):
    """What the traveler knows, which is usually very little. That is the point of Wander."""

    vibe: str = Field(description="Free text: mood, feeling, or fragments of what they want.")
    days: int = 5
    budget: Literal["shoestring", "comfortable", "splurge"] = "comfortable"
    travelling_from: str | None = None
    companions: str | None = None
    avoid: list[str] = Field(default_factory=list)


class Feedback(BaseModel):
    """How the traveler reacted to a round of postcards. Drives the narrowing loop."""

    liked: list[str] = Field(default_factory=list, description="Names of destinations they loved.")
    disliked: list[str] = Field(default_factory=list, description="Names of destinations they rejected.")
    note: str | None = Field(default=None, description="Free text such as 'more water, fewer crowds'.")


class Destination(BaseModel):
    name: str = Field(description="Place name, specific enough to look up (city, island, region).")
    country: str
    tagline: str = Field(description="At most 8 words, evocative, no cliches like 'hidden gem'.")
    why_it_fits: str = Field(description="Two sentences tying the place to the traveler's vibe.")
    visual_scene: str = Field(
        description=(
            "One vivid paragraph describing a single postcard scene for this place: what is in "
            "frame, the time of day and the light. Concrete objects, no text or signage."
        )
    )
    highlights: list[str] = Field(description="Exactly three concrete things to do or see.")
    best_months: str
    daily_budget_usd_low: int
    daily_budget_usd_high: int
    mood_tags: list[str]


class DestinationSet(BaseModel):
    destinations: list[Destination]
