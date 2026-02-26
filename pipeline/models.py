from pydantic import BaseModel, Field
from typing import List, Optional

class Location(BaseModel):
    port: Optional[str] = Field(None, description="The name of the port where the incident occurred, if any.")
    country: Optional[str] = Field(None, description="The country where the incident occurred.")
    lat: Optional[float] = Field(None, description="Latitude of the incident, if specified.")
    lon: Optional[float] = Field(None, description="Longitude of the incident, if specified.")

class MaritimeEvent(BaseModel):
    event_id: str = Field(..., description="A unique identifier for the event, e.g., EVT-<DATE>-<NUM>")
    event_date: str = Field(..., description="The date of the event in YYYY-MM-DD format.")
    location: Location
    vessels_involved: List[str] = Field(default_factory=list, description="List of normalized vessel names involved.")
    organizations_involved: List[str] = Field(default_factory=list, description="List of organizations (e.g., shipping companies, authorities) involved.")
    incident_type: str = Field(..., description="The classification of the incident (e.g., Collision, Piracy, Sanctions).")
    casualties: str = Field(..., description="The number or description of casualties or injuries. '0' if none reported.")
    cargo_type: Optional[str] = Field(None, description="Type of cargo involved, if mentioned.")
    summary: str = Field(..., description="A concise, 1-2 sentence executive summary of the event.")
    confidence_score: float = Field(..., description="Confidence score of the extraction from 0.0 to 1.0.")

class EventList(BaseModel):
    events: List[MaritimeEvent] = Field(default_factory=list, description="A list of maritime events extracted from the article.")
