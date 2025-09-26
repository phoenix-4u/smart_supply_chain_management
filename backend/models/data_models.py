#File: backend/models/data_models.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Enums for status and categories
class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProductCategory(str, Enum):
    PHARMACEUTICALS = "pharmaceuticals"
    MEDICAL_SUPPLIES = "medical_supplies"
    PPE = "ppe"
    SURGICAL_INSTRUMENTS = "surgical_instruments"
    DIAGNOSTIC_EQUIPMENT = "diagnostic_equipment"
    HOSPITAL_EQUIPMENT = "hospital_equipment"
    HOME_HEALTHCARE = "home_healthcare"
    DIGITAL_HEALTH = "digital_health"

class Region(str, Enum):
    NORTHEAST = "northeast"
    SOUTHEAST = "southeast"
    MIDWEST = "midwest"
    SOUTHWEST = "southwest"
    WEST_COAST = "west_coast"
    MOUNTAIN_STATES = "mountain_states"

class OrderStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = "success"
    message: Optional[str] = None

# Market Intelligence Models
class SocialMediaMetrics(BaseModel):
    """Social media monitoring metrics"""
    mentions_7d: int = Field(..., description="Mentions in last 7 days")
    mentions_change_pct: float = Field(..., description="Percentage change from previous period")
    sentiment_positive: float = Field(..., ge=0, le=1, description="Positive sentiment ratio")
    sentiment_negative: float = Field(..., ge=0, le=1, description="Negative sentiment ratio")
    sentiment_neutral: float = Field(..., ge=0, le=1, description="Neutral sentiment ratio")
    share_of_voice: Dict[str, float] = Field(..., description="Share of voice by competitor")
    trending_topics: List[str] = Field(default_factory=list)
    geographical_hotspots: Dict[str, int] = Field(default_factory=dict)
    influencer_mentions: List[Dict[str, Any]] = Field(default_factory=list)

class CompetitorData(BaseModel):
    """Competitor tracking data"""
    competitor_name: str
    price_changes: List[Dict[str, Any]] = Field(default_factory=list)
    product_launches: List[Dict[str, Any]] = Field(default_factory=list)
    market_share: float = Field(..., ge=0, le=1)
    recent_news: List[str] = Field(default_factory=list)
    supply_chain_status: str = "normal"

class WeatherAlert(BaseModel):
    """Weather and disaster alerts"""
    alert_id: str
    alert_type: str  # hurricane, flood, earthquake, etc.
    severity: AlertSeverity
    affected_regions: List[Region]
    population_affected: int
    healthcare_facilities_affected: int
    expected_duration_hours: int
    demand_impact_forecast: Dict[str, float]

class MarketIntelligenceResponse(BaseResponse):
    """Market intelligence comprehensive response"""
    social_media: SocialMediaMetrics
    competitors: List[CompetitorData]
    weather_alerts: List[WeatherAlert]
    economic_indicators: Dict[str, float]
    healthcare_utilization: Dict[str, Any]
    pricing_intelligence: Dict[str, Any]
    active_triggers: List[Dict[str, Any]]

# Demand Prediction Models
class DemandForecast(BaseModel):
    """Individual demand forecast"""
    product_category: ProductCategory
    region: Region
    forecast_date: datetime
    predicted_demand: float = Field(..., gt=0)
    confidence_interval_lower: float
    confidence_interval_upper: float
    factors_contributing: List[str] = Field(default_factory=list)
    accuracy_score: Optional[float] = Field(None, ge=0, le=1)

class DemandForecastResponse(BaseResponse):
    """Demand forecast response"""
    model_config = {'protected_namespaces': ()}

    forecasts: List[DemandForecast]
    model_performance: Dict[str, float]
    last_updated: datetime
    next_update: datetime

# Inventory Management Models
class InventoryItem(BaseModel):
    """Individual inventory item"""
    product_id: str
    product_name: str
    product_category: ProductCategory
    location: str
    current_stock: int = Field(..., ge=0)
    safety_stock_level: int = Field(..., ge=0)
    reorder_point: int = Field(..., ge=0)
    max_stock_level: int = Field(..., gt=0)
    average_daily_usage: float = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=0)
    unit_cost: float = Field(..., gt=0)
    total_value: float = Field(..., ge=0)

class InventoryRecommendation(BaseModel):
    """Inventory optimization recommendation"""
    product_id: str
    current_stock: int
    recommended_order_quantity: int = Field(..., ge=0)
    urgency: AlertSeverity
    reason: str
    cost_impact: float
    expected_stockout_date: Optional[datetime] = None

class InventoryStatus(BaseResponse):
    """Inventory status response"""
    items: List[InventoryItem]
    recommendations: List[InventoryRecommendation]
    total_inventory_value: float
    items_below_safety_stock: int
    items_requiring_reorder: int

# Supplier Management Models
class SupplierMetrics(BaseModel):
    """Supplier performance metrics"""
    on_time_delivery_rate: float = Field(..., ge=0, le=1)
    quality_score: float = Field(..., ge=0, le=5)
    average_lead_time_days: float = Field(..., ge=0)
    cost_competitiveness: float = Field(..., ge=0, le=1)
    reliability_score: float = Field(..., ge=0, le=1)
    communication_score: float = Field(..., ge=0, le=5)

class Supplier(BaseModel):
    """Supplier information"""
    supplier_id: str
    supplier_name: str
    contact_info: Dict[str, str]
    product_categories: List[ProductCategory]
    regions_served: List[Region]
    contract_start_date: datetime
    contract_end_date: datetime
    metrics: SupplierMetrics
    risk_level: AlertSeverity = AlertSeverity.LOW

class PurchaseOrder(BaseModel):
    """Purchase order details"""
    order_id: str
    supplier_id: str
    supplier_name: str
    order_date: datetime
    expected_delivery_date: datetime
    actual_delivery_date: Optional[datetime] = None
    status: OrderStatus
    items: List[Dict[str, Any]]  # product_id, quantity, unit_price
    total_amount: float = Field(..., gt=0)
    notes: Optional[str] = None

class SupplierPerformance(BaseResponse):
    """Supplier performance response"""
    suppliers: List[Supplier]
    purchase_orders: List[PurchaseOrder]
    performance_summary: Dict[str, float]

# System Status Models
class AgentStatusInfo(BaseModel):
    """Individual agent status information"""
    status: AgentStatus
    last_update: datetime
    error_message: Optional[str] = None
    uptime_percentage: float = Field(..., ge=0, le=100)
    tasks_completed_today: int = Field(..., ge=0)

class SystemStatus(BaseResponse):
    """Overall system status"""
    agents: Dict[str, AgentStatusInfo]
    overall_status: str
    database_status: str = "connected"
    api_response_time_ms: float = Field(..., gt=0)
    memory_usage_mb: float = Field(..., gt=0)
    cpu_usage_percentage: float = Field(..., ge=0, le=100)

# Analytics and KPI Models
class KPIMetrics(BaseModel):
    """Key Performance Indicators"""
    forecast_accuracy: float = Field(..., ge=0, le=1)
    inventory_turnover: float = Field(..., gt=0)
    stockout_rate: float = Field(..., ge=0, le=1)
    supplier_performance_avg: float = Field(..., ge=0, le=1)
    cost_savings_mtd: float = Field(..., ge=0)
    order_fill_rate: float = Field(..., ge=0, le=1)
    emergency_orders_count: int = Field(..., ge=0)

class DashboardData(BaseResponse):
    """Comprehensive dashboard data"""
    kpis: KPIMetrics
    market_intelligence_summary: Dict[str, Any]
    demand_forecast_summary: Dict[str, Any]
    inventory_summary: Dict[str, Any]
    supplier_summary: Dict[str, Any]
    recent_alerts: List[Dict[str, Any]]
    trending_products: List[Dict[str, Any]]

# Alert and Notification Models
class Alert(BaseModel):
    """System alert model"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    source: str  # market_intelligence, demand_prediction, etc.
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Configuration Models
class AgentConfiguration(BaseModel):
    """Agent configuration settings"""
    agent_name: str
    enabled: bool = True
    update_interval_seconds: int = Field(..., gt=0)
    configuration: Dict[str, Any] = Field(default_factory=dict)

# Validation functions
@validator('sentiment_positive', 'sentiment_negative', 'sentiment_neutral', pre=True)
def validate_sentiment_sum(cls, v, values):
    """Ensure sentiment values sum to approximately 1"""
    if 'sentiment_positive' in values and 'sentiment_negative' in values:
        total = values['sentiment_positive'] + values['sentiment_negative'] + v
        if abs(total - 1.0) > 0.01:  # Allow small floating point differences
            raise ValueError('Sentiment values must sum to 1.0')
    return v

# Helper functions for model creation
def create_sample_market_intelligence() -> MarketIntelligenceResponse:
    """Create sample market intelligence data for testing"""
    return MarketIntelligenceResponse(
        social_media=SocialMediaMetrics(
            mentions_7d=6550,
            mentions_change_pct=32.0,
            sentiment_positive=0.67,
            sentiment_negative=0.12,
            sentiment_neutral=0.21,
            share_of_voice={"Health First": 0.40, "Cardinal Health": 0.25, "Cencora": 0.20, "Others": 0.15},
            trending_topics=["#PPEshortage", "#SeasonalAllergies", "#Telehealth", "#FluSeason2024"],
            geographical_hotspots={"Midwest": 120, "Southeast": 80, "Northeast": 60}
        ),
        competitors=[
            CompetitorData(
                competitor_name="Cardinal Health",
                market_share=0.25,
                supply_chain_status="disrupted",
                recent_news=["Announced 8% price cut on surgical gloves in Midwest region"]
            )
        ],
        weather_alerts=[],
        economic_indicators={"unemployment_rate": 3.7, "healthcare_spending_growth": 4.2},
        healthcare_utilization={"hospital_occupancy": 0.85, "icu_utilization": 0.78},
        pricing_intelligence={"avg_price_change": -0.03, "competitor_undercutting": 0.15},
        active_triggers=[
            {"type": "social_media_spike", "product": "PPE", "region": "Midwest", "severity": "high"}
        ]
    )

def create_sample_demand_forecast() -> DemandForecastResponse:
    """Create sample demand forecast data for testing"""
    return DemandForecastResponse(
        forecasts=[
            DemandForecast(
                product_category=ProductCategory.PPE,
                region=Region.MIDWEST,
                forecast_date=datetime.now(),
                predicted_demand=1500.0,
                confidence_interval_lower=1200.0,
                confidence_interval_upper=1800.0,
                factors_contributing=["Social media spike", "Weather alert", "Seasonal trend"],
                accuracy_score=0.87
            )
        ],
        model_performance={"mae": 0.12, "mse": 0.08, "r2": 0.89},
        last_updated=datetime.now(),
        next_update=datetime.now()
    )
