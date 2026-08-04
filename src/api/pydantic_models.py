"""Pydantic schemas for request and response validation."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Customer-level features expected by the trained model pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CustomerId_4406",
                "Recency": 15.0,
                "Frequency": 5.0,
                "MonetarySum": 1200.5,
                "MonetaryMean": 240.1,
                "MonetaryStd": 50.2,
                "TransactionHour": 14.0,
                "TransactionDay": 12.0,
                "TransactionMonth": 6.0,
                "ProductCategory": "airtime",
                "ChannelId": "ChannelId_3",
                "PricingStrategy": "2",
            }
        }
    )

    customer_id: str = Field(..., description="Unique customer identifier.")
    Recency: float = Field(..., ge=0, description="Days since the customer's last transaction.")
    Frequency: float = Field(..., gt=0, description="Number of transactions.")
    MonetarySum: float = Field(..., description="Total transaction value.")
    MonetaryMean: float = Field(..., description="Average transaction value.")
    MonetaryStd: float = Field(..., ge=0, description="Std deviation of transaction values.")
    TransactionHour: float = Field(..., ge=0, le=23, description="Average transaction hour.")
    TransactionDay: float = Field(..., ge=1, le=31, description="Average day of month.")
    TransactionMonth: float = Field(..., ge=1, le=12, description="Average transaction month.")
    ProductCategory: str = Field(..., description="Most frequent product category.")
    ChannelId: str = Field(..., description="Most frequent channel.")
    PricingStrategy: str = Field(..., description="Most frequent pricing strategy.")


class PredictionResult(BaseModel):
    """Risk assessment returned for a single customer."""

    customer_id: str
    risk_probability: float = Field(..., ge=0, le=1)
    risk_category: str = Field(..., description="One of Low, Medium, or High.")
    credit_score: int = Field(..., ge=300, le=850)
    recommended_limit: Optional[float] = None
    recommended_term: Optional[int] = Field(default=None, description="Loan term in months.")


class BatchPredictionRequest(BaseModel):
    """Wrapper for scoring several customers in one request."""

    customers: List[CustomerFeatures] = Field(..., min_length=1)


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    count: int
