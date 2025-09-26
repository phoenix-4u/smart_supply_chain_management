# file: backend/main.py

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uvicorn

from config.settings import Settings
from models.data_models import (
    MarketIntelligenceResponse, DemandForecastResponse, 
    InventoryStatus, SupplierPerformance, SystemStatus
)
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.demand_prediction_agent import DemandPredictionAgent
from agents.inventory_optimization_agent import InventoryOptimizationAgent
from agents.supplier_coordination_agent import SupplierCoordinationAgent
from utils.data_processing import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Health First AI Supply Chain Management",
    description="Agentic AI system for healthcare supply chain optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
market_intelligence_agent = MarketIntelligenceAgent()
demand_prediction_agent = DemandPredictionAgent()
inventory_optimization_agent = InventoryOptimizationAgent()
supplier_coordination_agent = SupplierCoordinationAgent()
data_processor = DataProcessor()

# Global state to track agent statuses
agent_status = {
    "market_intelligence": {"status": "active", "last_update": datetime.now()},
    "demand_prediction": {"status": "active", "last_update": datetime.now()},
    "inventory_optimization": {"status": "active", "last_update": datetime.now()},
    "supplier_coordination": {"status": "active", "last_update": datetime.now()}
}

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Health First AI Supply Chain Management System")

    # Initialize database connections
    await data_processor.initialize_database()

    # Start background tasks for agents
    asyncio.create_task(run_market_intelligence_agent())
    asyncio.create_task(run_demand_prediction_agent())
    asyncio.create_task(run_inventory_optimization_agent())
    asyncio.create_task(run_supplier_coordination_agent())

    logger.info("All agents initialized and running")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Health First AI Supply Chain Management System")

    # Cleanup database connections
    await data_processor.close_database()

# Background tasks for agents
async def run_market_intelligence_agent():
    """Background task for market intelligence agent"""
    while True:
        try:
            await market_intelligence_agent.update_intelligence()
            agent_status["market_intelligence"]["last_update"] = datetime.now()
            agent_status["market_intelligence"]["status"] = "active"
        except Exception as e:
            logger.error(f"Market Intelligence Agent error: {str(e)}")
            agent_status["market_intelligence"]["status"] = "error"

        await asyncio.sleep(settings.MARKET_INTELLIGENCE_UPDATE_INTERVAL)

async def run_demand_prediction_agent():
    """Background task for demand prediction agent"""
    while True:
        try:
            await demand_prediction_agent.update_forecasts()
            agent_status["demand_prediction"]["last_update"] = datetime.now()
            agent_status["demand_prediction"]["status"] = "active"
        except Exception as e:
            logger.error(f"Demand Prediction Agent error: {str(e)}")
            agent_status["demand_prediction"]["status"] = "error"

        await asyncio.sleep(settings.DEMAND_FORECAST_UPDATE_INTERVAL)

async def run_inventory_optimization_agent():
    """Background task for inventory optimization agent"""
    while True:
        try:
            await inventory_optimization_agent.optimize_inventory()
            agent_status["inventory_optimization"]["last_update"] = datetime.now()
            agent_status["inventory_optimization"]["status"] = "active"
        except Exception as e:
            logger.error(f"Inventory Optimization Agent error: {str(e)}")
            agent_status["inventory_optimization"]["status"] = "error"

        await asyncio.sleep(settings.INVENTORY_CHECK_INTERVAL)

async def run_supplier_coordination_agent():
    """Background task for supplier coordination agent"""
    while True:
        try:
            await supplier_coordination_agent.coordinate_suppliers()
            agent_status["supplier_coordination"]["last_update"] = datetime.now()
            agent_status["supplier_coordination"]["status"] = "active"
        except Exception as e:
            logger.error(f"Supplier Coordination Agent error: {str(e)}")
            agent_status["supplier_coordination"]["status"] = "error"

        await asyncio.sleep(settings.SUPPLIER_COORDINATION_INTERVAL)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# System status endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status"""
    import random
    from models.data_models import AgentStatusInfo, AgentStatus

    # Convert agent_status dict to required format with missing fields
    agents_formatted = {}
    for agent_name, agent_data in agent_status.items():
        agents_formatted[agent_name] = AgentStatusInfo(
            status=AgentStatus(agent_data["status"]),
            last_update=agent_data["last_update"],
            uptime_percentage=random.uniform(95.0, 99.9),  # Simulated values
            tasks_completed_today=random.randint(50, 200)  # Simulated values
        )

    return SystemStatus(
        agents=agents_formatted,
        timestamp=datetime.now(),
        overall_status="healthy" if all(
            agent["status"] == "active" for agent in agent_status.values()
        ) else "degraded",
        api_response_time_ms=random.uniform(1.5, 50.0),  # Simulated values
        memory_usage_mb=random.uniform(150.0, 800.0),
        cpu_usage_percentage=random.uniform(5.0, 25.0)
    )

# Market Intelligence APIs
@app.get("/api/market-intelligence/social-media")
async def get_social_media_intelligence():
    """Get social media monitoring data"""
    try:
        data = await market_intelligence_agent.get_social_media_data()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Social media intelligence error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-intelligence/competitors")
async def get_competitor_intelligence():
    """Get competitor tracking data"""
    try:
        data = await market_intelligence_agent.get_competitor_data()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Competitor intelligence error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-intelligence/alerts")
async def get_market_alerts():
    """Get active market intelligence alerts"""
    try:
        alerts = await market_intelligence_agent.get_active_alerts()
        return JSONResponse(content=alerts)
    except Exception as e:
        logger.error(f"Market alerts error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-intelligence/summary", response_model=MarketIntelligenceResponse)
async def get_market_intelligence_summary():
    """Get comprehensive market intelligence summary"""
    try:
        summary = await market_intelligence_agent.get_summary()
        return summary
    except Exception as e:
        logger.error(f"Market intelligence summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Demand Prediction APIs
@app.get("/api/demand/forecast", response_model=DemandForecastResponse)
async def get_demand_forecast(
    product_category: Optional[str] = None,
    region: Optional[str] = None,
    days_ahead: int = 30
):
    """Get demand forecasts"""
    try:
        forecast = await demand_prediction_agent.get_forecast(
            product_category=product_category,
            region=region,
            days_ahead=days_ahead
        )
        return forecast
    except Exception as e:
        logger.error(f"Demand forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/demand/trigger")
async def trigger_demand_recalculation(background_tasks: BackgroundTasks):
    """Trigger immediate demand forecast recalculation"""
    try:
        background_tasks.add_task(demand_prediction_agent.recalculate_forecasts)
        return {"message": "Demand recalculation triggered", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Demand recalculation trigger error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/demand/accuracy")
async def get_forecast_accuracy():
    """Get demand forecast accuracy metrics"""
    try:
        accuracy = await demand_prediction_agent.get_accuracy_metrics()
        return JSONResponse(content=accuracy)
    except Exception as e:
        logger.error(f"Forecast accuracy error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Inventory Optimization APIs
@app.get("/api/inventory/levels", response_model=InventoryStatus)
async def get_inventory_levels(
    product_category: Optional[str] = None,
    location: Optional[str] = None
):
    """Get current inventory levels"""
    try:
        # Fetch inventory items
        inventory_items = await inventory_optimization_agent.get_inventory_levels(
            product_category=product_category,
            location=location
        )

        # Fetch recommendations and summary data
        recommendations_data = await inventory_optimization_agent.get_recommendations()

        # Construct the response
        inventory_status = InventoryStatus(
            items=inventory_items,
            recommendations=recommendations_data.get('recommendations', []),
            total_inventory_value=recommendations_data.get('summary', {}).get('total_inventory_value', 0),
            items_below_safety_stock=recommendations_data.get('summary', {}).get('items_below_safety_stock', 0),
            items_requiring_reorder=recommendations_data.get('summary', {}).get('items_requiring_reorder', 0),
            timestamp=datetime.now(),
            status="success"
        )

        return inventory_status
    except Exception as e:
        logger.error(f"Inventory levels error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/inventory/recommendations")
async def get_inventory_recommendations():
    """Get inventory optimization recommendations"""
    try:
        recommendations = await inventory_optimization_agent.get_recommendations()
        return JSONResponse(content=recommendations)
    except Exception as e:
        logger.error(f"Inventory recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inventory/adjust")
async def adjust_inventory_parameters(adjustment_data: dict):
    """Adjust inventory optimization parameters"""
    try:
        result = await inventory_optimization_agent.adjust_parameters(adjustment_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Inventory adjustment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Supplier Coordination APIs
@app.get("/api/suppliers/performance", response_model=SupplierPerformance)
async def get_supplier_performance():
    """Get supplier performance metrics"""
    try:
        # Fetch supplier data
        suppliers = await supplier_coordination_agent.get_supplier_performance()

        # Fetch purchase orders
        purchase_orders = await supplier_coordination_agent.get_purchase_orders()

        # Calculate performance summary
        total_suppliers = len(suppliers)
        if total_suppliers > 0:
            performance_summary = {
                "total_suppliers": total_suppliers,
                "average_on_time_delivery_rate": sum(s.metrics.on_time_delivery_rate for s in suppliers) / total_suppliers,
                "average_quality_score": sum(s.metrics.quality_score for s in suppliers) / total_suppliers,
                "average_lead_time_days": sum(s.metrics.average_lead_time_days for s in suppliers) / total_suppliers,
            }
        else:
            performance_summary = {
                "total_suppliers": 0,
                "average_on_time_delivery_rate": 0,
                "average_quality_score": 0,
                "average_lead_time_days": 0,
            }

        # Construct the response
        supplier_performance = SupplierPerformance(
            suppliers=suppliers,
            purchase_orders=purchase_orders,
            performance_summary=performance_summary,
            timestamp=datetime.now(),
            status="success"
        )

        return supplier_performance
    except Exception as e:
        logger.error(f"Supplier performance error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suppliers/orders")
async def get_purchase_orders(status: Optional[str] = None):
    """Get purchase order status"""
    try:
        orders = await supplier_coordination_agent.get_purchase_orders(status=status)
        return JSONResponse(content=orders)
    except Exception as e:
        logger.error(f"Purchase orders error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/suppliers/order")
async def create_purchase_order(order_data: dict):
    """Create new purchase order"""
    try:
        result = await supplier_coordination_agent.create_purchase_order(order_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Purchase order creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Reporting APIs
@app.get("/api/analytics/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        dashboard_data = {
            "market_intelligence": await market_intelligence_agent.get_summary(),
            "demand_forecast": await demand_prediction_agent.get_forecast(),
            "inventory_status": await inventory_optimization_agent.get_inventory_levels(),
            "supplier_performance": await supplier_coordination_agent.get_supplier_performance(),
            "system_status": agent_status,
            "timestamp": datetime.now()
        }
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Dashboard data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/kpis")
async def get_kpis():
    """Get key performance indicators"""
    try:
        kpis = await data_processor.calculate_kpis()
        return JSONResponse(content=kpis)
    except Exception as e:
        logger.error(f"KPIs calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
