#File: backend/config/settings.py

import os
from typing import Optional, List, Dict
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configuration"""

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    DEBUG: bool = Field(default=True, env="DEBUG")

    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./health_first.db", env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")

    # Agent Update Intervals (in seconds)
    MARKET_INTELLIGENCE_UPDATE_INTERVAL: int = Field(default=300, env="MARKET_INTELLIGENCE_UPDATE_INTERVAL")  # 5 minutes
    DEMAND_FORECAST_UPDATE_INTERVAL: int = Field(default=600, env="DEMAND_FORECAST_UPDATE_INTERVAL")  # 10 minutes
    INVENTORY_CHECK_INTERVAL: int = Field(default=900, env="INVENTORY_CHECK_INTERVAL")  # 15 minutes
    SUPPLIER_COORDINATION_INTERVAL: int = Field(default=1800, env="SUPPLIER_COORDINATION_INTERVAL")  # 30 minutes

    # External API Keys
    SOCIAL_MEDIA_API_KEY: Optional[str] = Field(default=None, env="SOCIAL_MEDIA_API_KEY")
    TWITTER_BEARER_TOKEN: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    NEWS_API_KEY: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    WEATHER_API_KEY: Optional[str] = Field(default=None, env="WEATHER_API_KEY")
    ECONOMIC_DATA_API_KEY: Optional[str] = Field(default=None, env="ECONOMIC_DATA_API_KEY")

    # Security Settings
    SECRET_KEY: str = Field(default="health-first-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")

    # Business Logic Configuration
    HEALTH_FIRST_COMPETITORS: List[str] = Field(default=[
        "Cardinal Health",
        "Cencora",  # formerly AmerisourceBergen
        "Owens & Minor",
        "Henry Schein",
        "CVS Health",
        "Walgreens Boots Alliance",
        "Medline Industries",
        "Patterson Companies"
    ], env=False)

    # Product Categories
    PRODUCT_CATEGORIES: List[str] = Field(default=[
        "Pharmaceuticals",
        "Medical Supplies",
        "PPE",
        "Surgical Instruments",
        "Diagnostic Equipment",
        "Hospital Equipment",
        "Home Healthcare",
        "Digital Health Solutions"
    ], env=False)

    # Geographic Regions
    SERVICE_REGIONS: List[str] = Field(default=[
        "Northeast",
        "Southeast",
        "Midwest",
        "Southwest",
        "West Coast",
        "Mountain States"
    ], env=False)

    # Market Intelligence Thresholds
    SOCIAL_MEDIA_SPIKE_THRESHOLD: float = Field(default=2.0, env="SOCIAL_MEDIA_SPIKE_THRESHOLD")  # 2x normal volume
    SENTIMENT_CHANGE_THRESHOLD: float = Field(default=0.2, env="SENTIMENT_CHANGE_THRESHOLD")  # 20% change
    COMPETITOR_PRICE_CHANGE_THRESHOLD: float = Field(default=0.05, env="COMPETITOR_PRICE_CHANGE_THRESHOLD")  # 5% change

    # Demand Prediction Parameters
    FORECAST_HORIZON_DAYS: int = Field(default=90, env="FORECAST_HORIZON_DAYS")
    CONFIDENCE_INTERVAL: float = Field(default=0.95, env="CONFIDENCE_INTERVAL")
    MIN_HISTORICAL_DATA_POINTS: int = Field(default=30, env="MIN_HISTORICAL_DATA_POINTS")

    # Inventory Optimization Parameters
    SAFETY_STOCK_MULTIPLIER: float = Field(default=1.5, env="SAFETY_STOCK_MULTIPLIER")
    REORDER_POINT_BUFFER: float = Field(default=0.2, env="REORDER_POINT_BUFFER")  # 20% buffer
    MAX_INVENTORY_LEVEL_MULTIPLIER: float = Field(default=3.0, env="MAX_INVENTORY_LEVEL_MULTIPLIER")

    # Supplier Coordination Parameters
    SUPPLIER_PERFORMANCE_THRESHOLD: float = Field(default=0.95, env="SUPPLIER_PERFORMANCE_THRESHOLD")  # 95% performance
    DELIVERY_TIME_TOLERANCE_DAYS: int = Field(default=2, env="DELIVERY_TIME_TOLERANCE_DAYS")
    QUALITY_SCORE_THRESHOLD: float = Field(default=4.0, env="QUALITY_SCORE_THRESHOLD")  # Out of 5

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="health_first.log", env="LOG_FILE")

    # Cache Configuration
    CACHE_TTL: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")

    # Monitoring and Alerting
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    ALERT_EMAIL_ENABLED: bool = Field(default=False, env="ALERT_EMAIL_ENABLED")
    ALERT_EMAIL_RECIPIENTS: str = Field(default="", env="ALERT_EMAIL_RECIPIENTS")

    # Machine Learning Model Configuration
    MODEL_RETRAIN_INTERVAL_HOURS: int = Field(default=24, env="MODEL_RETRAIN_INTERVAL_HOURS")
    MODEL_ACCURACY_THRESHOLD: float = Field(default=0.85, env="MODEL_ACCURACY_THRESHOLD")
    ENSEMBLE_MODEL_WEIGHTS: Dict[str, float] = Field(default={
        "linear_regression": 0.3,
        "random_forest": 0.4,
        "xgboost": 0.3
    })

    # Data Processing Configuration
    DATA_PROCESSING_BATCH_SIZE: int = Field(default=1000, env="DATA_PROCESSING_BATCH_SIZE")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Configuration validation
def validate_configuration():
    """Validate configuration settings"""
    required_settings = []

    # Check for required external API keys if features are enabled
    if settings.SOCIAL_MEDIA_API_KEY is None:
        print("Warning: SOCIAL_MEDIA_API_KEY not set. Social media monitoring will use mock data.")

    if settings.NEWS_API_KEY is None:
        print("Warning: NEWS_API_KEY not set. News monitoring will use mock data.")

    if settings.WEATHER_API_KEY is None:
        print("Warning: WEATHER_API_KEY not set. Weather monitoring will use mock data.")

    # Validate thresholds
    assert 0 < settings.CONFIDENCE_INTERVAL < 1, "Confidence interval must be between 0 and 1"
    assert settings.SAFETY_STOCK_MULTIPLIER > 0, "Safety stock multiplier must be positive"
    assert settings.SUPPLIER_PERFORMANCE_THRESHOLD > 0, "Supplier performance threshold must be positive"

    print("✅ Configuration validation completed")

if __name__ == "__main__":
    validate_configuration()
    print("Settings loaded successfully")
    print(f"API will run on {settings.API_HOST}:{settings.API_PORT}")
    print(f"Database URL: {settings.DATABASE_URL}")
