from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    email: str
    name: str
    preferences: Optional[dict] = Field(default_factory=dict)
    style_profile: Optional[dict] = Field(default_factory=dict)

class ProductBase(BaseModel):
    name: str
    category: str
    style: str
    color: str
    occasion: List[str]
    season: List[str]
    price: float
    brand: str
    image_url: str
    description: Optional[str] = None
    attributes: Optional[dict] = Field(default_factory=dict)

class UserInteraction(BaseModel):
    user_id: str
    product_id: str
    interaction_type: str  # view, like, purchase
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Optional[dict] = Field(default_factory=dict)

class WeatherPreference(BaseModel):
    temperature_range: List[float]
    weather_conditions: List[str]
    suitable_categories: List[str]
    priority_score: float