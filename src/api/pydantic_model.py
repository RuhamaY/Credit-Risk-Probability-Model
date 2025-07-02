from pydantic import BaseModel, Field # Import Field for better examples/descriptions
from typing import Optional

class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="CUST_001", description="Unique identifier for the customer.")
    Recency: float = Field(..., example=15.0, description="Recency (days since last purchase).")
    Frequency: float = Field(..., example=5.0, description="Frequency (number of purchases).")
    MonetarySum: float = Field(..., example=120.50, description="Total monetary value of purchases.")
    MonetaryMean: float = Field(..., example=24.10, description="Average monetary value per purchase.")
    MonetaryStd: float = Field(..., example=5.20, description="Standard deviation of monetary values.")
    ProductCategory: str = Field(..., example="Electronics", description="Main product category purchased.")
    ChannelId: str = Field(..., example="MobileApp", description="Channel used for purchase.")
    PricingStrategy: str = Field(..., example="Standard", description="Pricing strategy applied.")


class PredictionResult(BaseModel):
    customer_id: str
    risk_probability: float
    risk_category: str
    credit_score: int
    recommended_limit: Optional[float] = None
    recommended_term: Optional[int] = None
